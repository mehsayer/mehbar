from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Coroutine, Iterable
from dataclasses import dataclass
from enum import Enum
from functools import cache, lru_cache
from typing import Any

import anyio
from gi.repository import GLib, Gtk, Pango  # type: ignore

from .actions import ActionInterface, CallableAction, ExecAction
from .exceptions import BarConfigError
from .resource_manager import ResourceManager
from .tools import OptionalFormatter, next_prime


class IconPosition(Enum):
    START = 0
    END = 1
    NONE = 2


@dataclass(frozen=True)
class WidgetContent:
    RE_SPEC = re.compile(
        r"""\[\s*
            (?:
                (?:icon\s*=\s*(?P<icon>[\w\!\_\:\/\.\-]+)\s*
                (?P<pos>[<>])?\s*(?:;)\s*)
                |(?:classes\s*:\s*(?:
                    (?:
                        \s*(?:,)?\s*label\s*=\s*(?P<lcls>[\w\_\-\!\ ]+))?
                        |(?:\s*(?:,)?\s*icon\s*=\s*(?P<icls>[\w\_\-\!\ ]+))?
                        |(?:\s*(?:,)?\s*widget\s*=\s*(?P<wcls>[\w\_\-\!\ ]+))
                    ?){1,3}\s*(?:;)\s*
                )
                |(?:\s*tooltip\s*=\s*
                    (?P<tooltip>[^;\]]+)?\s*(?:;)\s*)
            ){1,3}\s*
        \]""",
        re.X,
    )

    icon: str | None
    icon_position: IconPosition
    label: str | None
    tooltip_text: str | None
    icon_classes: frozenset[str] | None
    label_classes: frozenset[str] | None
    widget_classes: frozenset[str] | None

    @classmethod
    @cache
    def parse(cls, text: str | None) -> WidgetContent:
        """
        Parses the string of the following format, returns a `WidgetContent` instance:
            [<ICON SPECIFICATION>;<CSS CLASSES>;<TOOLTIP TEXT>;] <TEXT>
        Where:
            ICON SPECIFICATION
                icon=<ICON NAME><ICON POSITION>
                    ICON NAME
                        The name of the icon as specified in the configuration
                    ICON POSITION
                        `>`, to the right, or `<`, to the left from the label.
                        Optional, defaults to `<` if not specified.
            CSS CLASSES
                classes:widget=<CLASS 1> ... <CLASS N>, icon=..., label=...
                    widget=<CLASS 1> ... <CLASS N>
                        Space-separated list of CSS classes to be applied to the widget
                    icon=<CLASS 1> ... <CLASS N>
                        Space-separated list of CSS classes to be applied to the icon
                    label=<CLASS 1> ... <CLASS N>
                        Space-separated list of CSS classes to be applied to the label
            TOOLTIP
                tooltip=<TOOLTIP TEXT>
                    Tooltip text, may contain format fields
            TEXT
                Label text

            All fields are optional, empty (lacking values), repeating and unknown
            fields are not allowed
        """

        icon = None
        icon_position = IconPosition.NONE
        icon_classes = None
        label_classes = None
        widget_classes = None
        tooltip_text = None
        label = None

        if text is not None:
            if (match_spec := cls.RE_SPEC.search(text)) is not None:
                icon = match_spec.group("icon")

                if icon is not None:
                    if icon in ("!none", "!default"):
                        icon = None

                    match match_spec.group("pos"):
                        case ">":
                            icon_position = IconPosition.END
                        case "<" | None:
                            icon_position = IconPosition.START

                if (icon_classes_ := match_spec.group("icls")) is not None:
                    icon_classes = frozenset(icon_classes_.split())

                if (label_classes_ := match_spec.group("lcls")) is not None:
                    label_classes = frozenset(label_classes_.split())

                if (widget_classes_ := match_spec.group("wcls")) is not None:
                    widget_classes = frozenset(widget_classes_.split())

                if (tooltip_text_ := match_spec.group("tooltip")) is not None:
                    tooltip_text = tooltip_text_.strip()

            label = cls.RE_SPEC.sub("", text)

        return cls(
            icon,
            icon_position,
            label,
            tooltip_text,
            icon_classes,
            label_classes,
            widget_classes,
        )


class IdleUpdater:
    """Calls `apply` with the most recently submitted value on GTK main
    thread. Values superseded before GTK main thread catches up are dropped,
    as are values equal to the previously submitted one. `submit()` is safe
    to call from any thread."""

    _NOTHING = object()

    def __init__(self, apply: Callable[[Any], None]):
        self._apply = apply
        self._lock = threading.Lock()
        self._last: Any = self._NOTHING
        self._pending: Any = self._NOTHING

    def submit(self, value: Any):
        with self._lock:
            if value == self._last:
                return

            self._last = value
            scheduled = self._pending is not self._NOTHING
            self._pending = value

        if not scheduled:
            GLib.idle_add(self._flush)

    def _flush(self) -> bool:
        with self._lock:
            value, self._pending = self._pending, self._NOTHING

        if value is not self._NOTHING:
            self._apply(value)

        return GLib.SOURCE_REMOVE


class RewriteMixin:
    """Rewrites text using the `rewrite` mapping of regular expressions to
    replacements, the first matching expression wins."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        rules = self.cfg.get("rewrite") or {}  # type: ignore[attr-defined]

        if not isinstance(rules, dict):
            raise BarConfigError("'rewrite' must be a mapping")

        try:
            self._rewrite_rules = [(re.compile(p), r) for p, r in rules.items()]
        except re.error as ex:
            raise BarConfigError(f"invalid rewrite pattern: {ex}") from ex

        self.rewrite = lru_cache(maxsize=64)(self._rewrite)

    def _rewrite(self, text: str) -> str:
        if text is not None:
            for pattern, repl in self._rewrite_rules:
                if pattern.match(text) is not None:
                    try:
                        return pattern.sub(repl, text)
                    except re.error as ex:
                        logging.error("cannot rewrite '%s': %s", text, ex)
                        break
        return text


class JSONInputMixin:
    """Sets widget content from text lines. A line containing a JSON object
    provides format fields, `ramp_level` selects the ramp. Any other line is
    available as the `label` field."""

    def set_content_from_line_i(self, line: str):
        if not (line := line.strip()):
            return

        values = None

        if line.startswith("{"):
            try:
                values = json.loads(line)
            except json.JSONDecodeError as ex:
                logging.warning("failed to parse JSON input: %s", ex)

        if not isinstance(values, dict):
            values = {"label": line}

        # 'ramp_level' is also available as a label field
        ramp_level = values.get("ramp_level", -1)

        try:
            ramp_level = int(ramp_level)
        except (TypeError, ValueError, OverflowError):
            logging.warning("invalid 'ramp_level' value: %r", ramp_level)
            ramp_level = -1

        self.set_content_i(self.format_content(ramp_level, values))  # type: ignore[attr-defined]


class BarWidget(Gtk.Box):
    """Base class of all bar widgets.

    Widgets are constructed on GTK main thread, their `run()` coroutine runs
    on a separate event loop thread. Methods ending with `_i` are safe to call
    from any thread, they schedule GTK updates on the main thread.
    """

    TYPE: str
    UNIQUE = True
    STATIC = False
    # make the interval at least `ResourceManager.INTERVAL_OFFSET` apart from
    # other widget intervals
    RELAX_INTERVAL = True
    # wake up at wall clock multiples of the interval
    ALIGN_INTERVAL = False

    def __init__(
        self, name: str, res_mgr: ResourceManager, cfg: dict[str, Any] | None = None
    ):
        super().__init__()

        self.res_mgr = res_mgr
        # for use outside of GTK main thread
        self.widget_name = name
        self.set_name(name)
        self.add_css_class("bar-widget")

        self.cfg = res_mgr.get_cfg_for_name(name) if cfg is None else cfg

        interval = self.cfg.get("interval", 0)

        if not isinstance(interval, (int, float)) or interval < 0:
            raise BarConfigError("'interval' must be a non-negative number")

        if interval > 0:
            if self.RELAX_INTERVAL and self.cfg.get("relax_interval", True):
                interval = res_mgr.relax_interval(interval)

            if self.ALIGN_INTERVAL:
                self._period = float(interval)
            else:
                self._period = next_prime(int(interval * 1000)) / 1000
        else:
            self._period = 0.0

        self.interval = interval
        self._stopped = False
        self._deadline: float | None = None

        if (onclick := self.cfg.get("onclick")) is not None:
            if not isinstance(onclick, list):
                raise BarConfigError("'onclick' must be a list of commands")

            # the first command is for the primary (left) button
            for button, cmdline in enumerate(onclick, start=1):
                if cmdline:
                    self.onclick_exec(button, cmdline)

        if (onscroll := self.cfg.get("onscroll")) is not None:
            if not isinstance(onscroll, list) or len(onscroll) != 2:
                raise BarConfigError("'onscroll' must be a list of two commands")
            self.onscroll_exec(*onscroll)

    # Lifecycle

    async def sleep_interval(self) -> bool:
        """Returns immediately when called for the first time, then sleeps for
        the configured interval. Returns `False` if the widget should stop."""

        if self._stopped:
            return False

        if self._deadline is None:
            self._deadline = anyio.current_time()
            return True

        if self._period <= 0:
            return False

        if self.ALIGN_INTERVAL:
            # wake up just after the boundary, never before it
            await anyio.sleep(self._period - time.time() % self._period + 0.005)
        else:
            now = anyio.current_time()
            # do not try to catch up if the widget fell behind
            self._deadline = max(self._deadline + self._period, now)
            await anyio.sleep_until(self._deadline)

        return not self._stopped

    def shutdown(self):
        self._stopped = True

    async def run_wrapper(self):
        if not self.STATIC:
            await self.run()

    async def run(self):
        raise NotImplementedError()

    # Thread helpers

    def idle_add(self, func: Callable, *args: Any):
        """Calls `func` on GTK main thread."""

        def _run():
            func(*args)
            return GLib.SOURCE_REMOVE

        GLib.idle_add(_run)

    def call_soon(self, func: Callable, *args: Any):
        """Calls `func` on the widget event loop thread."""
        self.res_mgr.loop.call_soon(func, *args)

    def run_soon(self, coro_func: Callable[..., Coroutine], *args: Any):
        """Starts `coro_func` as a task on the widget event loop."""
        self.res_mgr.loop.run_soon(coro_func, *args)

    def set_visible_i(self, state: bool):
        self.idle_add(self.set_visible, state)

    # Input

    def _onclick(self, button: int, action: ActionInterface):
        if button >= 0 and action is not None:
            controller = Gtk.GestureClick.new()
            controller.set_button(button)
            controller.connect("pressed", lambda *_: action.run())
            self.add_controller(controller)

    def _onscroll(self, action_up: ActionInterface, action_down: ActionInterface):
        def _scroll(_ctrl, _dx: float, dy: float) -> bool:
            if dy < 0:
                action_up.run()
            elif dy > 0:
                action_down.run()
            return True

        if action_up is not None and action_down is not None:
            controller = Gtk.EventControllerScroll.new(
                Gtk.EventControllerScrollFlags.VERTICAL
                | Gtk.EventControllerScrollFlags.DISCRETE
            )
            controller.connect("scroll", _scroll)
            self.add_controller(controller)

    def onclick_call(self, button: int, func: Callable, *args, **kwargs):
        """`func` is called on GTK main thread."""
        self._onclick(button, CallableAction(func, *args, **kwargs))

    def onclick_exec(self, button: int, cmdline: str | list[str]):
        self._onclick(button, ExecAction(cmdline))

    def onscroll_call(self, func_up: Callable, func_down: Callable):
        """Functions are called on GTK main thread."""
        self._onscroll(CallableAction(func_up), CallableAction(func_down))

    def onscroll_exec(self, cmdline_up: str | list[str], cmdline_down: str | list[str]):
        self._onscroll(ExecAction(cmdline_up), ExecAction(cmdline_down))


class WidgetBase(BarWidget):
    """A widget showing a label and, optionally, an icon. The icon is laid out
    according to the first icon specification found in the label, the ramp
    or `layout_specs`, texts the widget may show otherwise."""

    DFL_RAMP_IDX = 100

    def __init__(
        self,
        name: str,
        res_mgr: ResourceManager,
        cfg: dict[str, Any] | None = None,
        layout_specs: Iterable[str] = (),
    ):
        super().__init__(name, res_mgr, cfg)

        self.label = Gtk.Label.new()
        self.label.add_css_class("bar-widget-label")

        self.icon = Gtk.Image.new()
        self.icon.add_css_class("bar-widget-icon")

        if not isinstance(label := self.cfg.get("label", ""), str):
            raise BarConfigError("'label' must be a string")

        self.content = WidgetContent.parse(label)

        self.ramp: list[str] = self.cfg.get("ramp") or []

        if not isinstance(self.ramp, list) or not all(
            isinstance(r, str) for r in self.ramp
        ):
            raise BarConfigError("'ramp' must be a list of strings")

        self.max_ramp_level = self.cfg.get("max_ramp_level", self.DFL_RAMP_IDX)

        icon_position = self.content.icon_position

        if icon_position == IconPosition.NONE:
            for ramp_str in (*self.ramp, *layout_specs):
                ramp_content = WidgetContent.parse(ramp_str)
                if ramp_content.icon_position != IconPosition.NONE:
                    icon_position = ramp_content.icon_position
                    break

        if icon_position == IconPosition.START:
            self.icon.add_css_class("bar-widget-icon-start")
            self.append(self.icon)
            self.append(self.label)
        elif icon_position == IconPosition.END:
            self.icon.add_css_class("bar-widget-icon-end")
            self.append(self.label)
            self.append(self.icon)
        else:
            self.append(self.label)

        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.CENTER)
        self.set_homogeneous(False)

        self.icon.set_pixel_size(self.res_mgr.pixel_size)

        self.formatter = OptionalFormatter()

        self.label.set_xalign(0.5)
        self.label.set_yalign(0.5)
        self.label.set_single_line_mode(True)

        if (width_chars := self.cfg.get("width_chars", 0)) > 0:
            self.label.set_width_chars(width_chars)

        if (max_width_chars := self.cfg.get("max_width_chars", 0)) > 0:
            self.label.set_max_width_chars(max_width_chars)
            self.label.set_ellipsize(Pango.EllipsizeMode.END)

        self._updater = IdleUpdater(self.apply_content)

        # State of GTK widgets, GTK main thread only
        self._shown_icon: str | None = None
        self._shown_label: str | None = None
        self._shown_tooltip: str | None = None
        self._shown_classes: dict[Gtk.Widget, frozenset[str]] = {
            self: frozenset(),
            self.label: frozenset(),
            self.icon: frozenset(),
        }

    @property
    def max_ramp_level(self) -> int:
        return self._max_ramp_level

    @max_ramp_level.setter
    def max_ramp_level(self, value: int | float):
        if not isinstance(value, (int, float)) or value < 1:
            raise BarConfigError("'max_ramp_level' must be a number greater than 0")
        self._max_ramp_level = value

    # Ramps

    def ramp_index(self, ramp_level: int) -> int | None:
        """Maps `ramp_level` to an index in `self.ramp`."""
        if ramp_level < 0 or not self.ramp:
            return None

        level = min(ramp_level, self.max_ramp_level - 1)
        return int(level / (self.max_ramp_level / len(self.ramp)))

    def get_ramp(self, ramp_level: int = -1) -> WidgetContent | None:
        if (idx := self.ramp_index(ramp_level)) is None:
            return None
        return WidgetContent.parse(self.ramp[idx])

    # Content

    def get_content(self, ramp_level: int = -1, **fields: Any) -> WidgetContent:
        return self.format_content(ramp_level, fields)

    def format_content(self, ramp_level: int, fields: dict[str, Any]) -> WidgetContent:
        """Formats the label using `fields` and picks the ramp entry for
        `ramp_level`, the `ramp` field is set to the ramp entry label."""
        base = self.content
        ramp = self.get_ramp(ramp_level)

        icon = base.icon
        tooltip = base.tooltip_text
        icon_classes = base.icon_classes
        label_classes = base.label_classes
        widget_classes = base.widget_classes

        if ramp is not None:
            fields = {"ramp": ramp.label} | fields

            if ramp.icon is not None:
                icon = ramp.icon
            if ramp.tooltip_text is not None:
                tooltip = ramp.tooltip_text
            if ramp.icon_classes is not None:
                icon_classes = ramp.icon_classes
            if ramp.label_classes is not None:
                label_classes = ramp.label_classes
            if ramp.widget_classes is not None:
                widget_classes = ramp.widget_classes

        label = self.formatter.vformat(base.label, (), fields).strip()

        if tooltip is not None:
            tooltip = self.formatter.vformat(tooltip, (), fields).strip() or None

        return WidgetContent(
            icon,
            base.icon_position,
            label,
            tooltip,
            icon_classes,
            label_classes,
            widget_classes,
        )

    def set_new_content_i(self, ramp_level: int = -1, **fields: Any):
        """Formats the label using `fields`, picks the ramp for `ramp_level`
        and schedules the widget update."""
        self.set_content_i(self.format_content(ramp_level, fields))

    def set_content_i(self, content: WidgetContent):
        """Schedules the widget update. Updates are coalesced: if GTK main
        thread did not catch up yet, only the latest content is shown."""
        self._updater.submit(content)

    def set_text_i(self, text: str):
        """Shows `text` as is, it may contain an icon specification."""
        self.set_content_i(WidgetContent.parse(text))

    def _sync_css_classes(self, widget: Gtk.Widget, classes: frozenset[str] | None):
        wanted = frozenset() if classes is None else classes - {"!none"}
        shown = self._shown_classes[widget]

        if wanted != shown:
            for class_name in shown - wanted:
                widget.remove_css_class(class_name)
            for class_name in wanted - shown:
                widget.add_css_class(class_name)
            self._shown_classes[widget] = wanted

    def apply_content(self, content: WidgetContent):
        """Updates GTK widgets, must be called from GTK main thread."""

        if content.icon is not None and content.icon != self._shown_icon:
            self._shown_icon = content.icon
            self.icon.set_from_paintable(self.res_mgr.get_paintable(content.icon))

        if content.label is not None and content.label != self._shown_label:
            self._shown_label = content.label
            self.label.set_label(content.label)

        if content.tooltip_text != self._shown_tooltip:
            self._shown_tooltip = content.tooltip_text
            self.set_tooltip_text(content.tooltip_text)

        self._sync_css_classes(self, content.widget_classes)
        self._sync_css_classes(self.label, content.label_classes)
        self._sync_css_classes(self.icon, content.icon_classes)
