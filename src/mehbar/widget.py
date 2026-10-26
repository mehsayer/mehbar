from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Coroutine, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache, wraps
from typing import Any

import anyio
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from gi.repository import GLib, Gtk, Pango  # type: ignore

from .actions import ActionInterface, CallableAction, ExecAction
from .exceptions import BarConfigError
from .resource_manager import EventLoopBridge, ResourceManager
from .tools import OptionalFormatter, describe_exception, next_prime, split_format


class IconPosition(Enum):
    START = 0
    END = 1
    NONE = 2
    # between two labels, only set by `{ramp}` in the middle of a label
    MIDDLE = 3


# hides the icon, unlike `!default` which keeps the label icon
NO_ICON = "!none"


@dataclass(frozen=True)
class WidgetContent:
    # a bracketed list of `;`-terminated fields, each starting with a known key
    RE_SPEC = re.compile(r"\[\s*((?:(?:icon|classes|tooltip)\s*[=:][^;\]]*;\s*)+)\]")
    RE_ICON = re.compile(r"icon\s*=\s*([\w!:/.\-]+)\s*([<>]?)")
    RE_CLASSES = re.compile(r"classes\s*(?:=\s*(?P<widget>.*)|:\s*(?P<parts>.*))")
    RE_CLASSES_PART = re.compile(r"(widget|icon|label)\s*=\s*(.*)")
    RE_CLASS_NAME = re.compile(r"-?[A-Za-z_][\w\-]*|!none")
    RE_TOOLTIP = re.compile(r"tooltip\s*=\s*(.*)")

    icon: str | None
    icon_position: IconPosition
    label: str | None
    tooltip_text: str | None
    icon_classes: frozenset[str] | None
    label_classes: frozenset[str] | None
    widget_classes: frozenset[str] | None
    # text after the icon, when it is between two labels
    label_end: str | None = None

    @classmethod
    def _parse_class_names(cls, text: str, spec: str) -> frozenset[str]:
        names = text.split()

        if not names or not all(cls.RE_CLASS_NAME.fullmatch(n) for n in names):
            raise BarConfigError(f"invalid CSS class names '{text}' in '{spec}'")

        return frozenset(names)

    @classmethod
    @lru_cache(maxsize=256)
    def parse(cls, text: str | None) -> WidgetContent:
        """
        Parses the string of the following format, returns a `WidgetContent` instance:
            [<ICON SPECIFICATION>;<CSS CLASSES>;<TOOLTIP TEXT>;] <TEXT>
        Where:
            ICON SPECIFICATION
                icon=<ICON NAME><ICON POSITION>
                    ICON NAME
                        The name of the icon as specified in the configuration,
                        `!none` hides the icon, `!default` keeps the label icon
                        (useful in ramps)
                    ICON POSITION
                        `>`, to the right, or `<`, to the left from the label.
                        Optional, defaults to `<` if not specified. A `{ramp}`
                        field in the label places ramp icons instead.
            CSS CLASSES
                classes:widget=<CLASS 1> ... <CLASS N>, icon=..., label=...
                    widget=<CLASS 1> ... <CLASS N>
                        Space-separated list of CSS classes to be applied to the widget
                    icon=<CLASS 1> ... <CLASS N>
                        Space-separated list of CSS classes to be applied to the icon
                    label=<CLASS 1> ... <CLASS N>
                        Space-separated list of CSS classes to be applied to the label
                classes=<CLASS 1> ... <CLASS N>
                    Same as `classes:widget=...`
                `!none` instead of class names removes the classes
            TOOLTIP
                tooltip=<TOOLTIP TEXT>
                    Tooltip text, may contain format fields, but not `;` or `]`
            TEXT
                Label text

            Fields may come in any order and are all optional. Each field must
            end with `;`. Empty, repeating and malformed fields, as well as
            more than one specification, raise `BarConfigError`. Brackets that
            do not start with a known field are left in the text as is.
        """

        icon = None
        icon_position = IconPosition.NONE
        classes: dict[str, frozenset[str]] = {}
        tooltip_text = None

        if text is None:
            return cls(None, icon_position, None, None, None, None, None)

        specs = list(cls.RE_SPEC.finditer(text))

        if len(specs) > 1:
            raise BarConfigError(f"more than one specification in '{text}'")

        if specs:
            seen = set()

            for field in specs[0].group(1).split(";")[:-1]:
                field = field.strip()
                key = field.partition("=")[0].partition(":")[0].strip()

                if key in seen:
                    raise BarConfigError(f"repeating '{key}' field in '{text}'")
                seen.add(key)

                if key == "icon":
                    if (m := cls.RE_ICON.fullmatch(field)) is None:
                        raise BarConfigError(f"invalid icon '{field}' in '{text}'")

                    icon = None if m.group(1) == "!default" else m.group(1)
                    icon_position = (
                        IconPosition.END if m.group(2) == ">" else IconPosition.START
                    )

                elif key == "classes":
                    if (m := cls.RE_CLASSES.fullmatch(field)) is None:
                        raise BarConfigError(f"invalid classes '{field}' in '{text}'")

                    if (widget := m.group("widget")) is not None:
                        classes["widget"] = cls._parse_class_names(widget, text)
                    else:
                        for part in m.group("parts").split(","):
                            m_part = cls.RE_CLASSES_PART.fullmatch(part.strip())

                            if m_part is None:
                                raise BarConfigError(
                                    f"invalid classes '{part.strip()}' in '{text}'"
                                )

                            target, names = m_part.groups()

                            if target in classes:
                                raise BarConfigError(
                                    f"repeating '{target}' classes in '{text}'"
                                )

                            classes[target] = cls._parse_class_names(names, text)

                else:  # tooltip, the only other key RE_SPEC accepts
                    m = cls.RE_TOOLTIP.fullmatch(field)

                    if m is None or not (tooltip_text := m.group(1).strip()):
                        raise BarConfigError(f"invalid tooltip '{field}' in '{text}'")

            text = text[: specs[0].start()] + text[specs[0].end() :]

        return cls(
            icon,
            icon_position,
            text,
            tooltip_text,
            classes.get("icon"),
            classes.get("label"),
            classes.get("widget"),
        )


def spread_index(level: float, max_level: float, count: int) -> int:
    """Maps `level` in 0..`max_level` to one of `count` equal ranges."""
    return int(min(max(level, 0), max_level - 1) / (max_level / count))


class RequestQueue:
    """Passes requests, e.g. of input handlers on GTK main thread, to the
    widget event loop. Requests are dropped while nothing receives them or
    while the queue is full."""

    def __init__(self, loop: EventLoopBridge, size: int = 8):
        self._loop = loop
        self._size = size
        self._send: ObjectSendStream | None = None

    def put(self, request: Any):
        """Safe to call from any thread."""
        self._loop.call_soon(self._put, request)

    def _put(self, request: Any):
        if self._send is not None:
            try:
                self._send.send_nowait(request)
            except (anyio.WouldBlock, anyio.ClosedResourceError):
                pass

    @contextmanager
    def receive(self) -> Iterator[ObjectReceiveStream]:
        """Yields the stream of requests, event loop thread only."""
        send, receive = anyio.create_memory_object_stream[Any](self._size)
        self._send = send

        try:
            yield receive
        finally:
            self._send = None
            send.close()
            receive.close()


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
    replacements. The first expression matching at the start of the text
    wins, the text it matches is replaced."""

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
                        # the first occurrence is the match at the start
                        return pattern.sub(repl, text, count=1)
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


class HorizontalScroller(Gtk.ScrolledWindow):
    """Shows `child` in at most `width` pixels, scrolled by the mouse wheel
    at `speed` pixels per step, or in its full width if `width` is 0.
    `guard` wraps the scroll handler, see `BarWidget.guard()`."""

    MIN_SPEED = 1
    MAX_SPEED = 100

    def __init__(self, child: Gtk.Widget, width: int, speed: float, guard: Callable):
        super().__init__()

        self.set_propagate_natural_height(True)
        self.set_has_frame(False)
        self.set_kinetic_scrolling(False)

        self.viewport: Gtk.Viewport | None = None

        if width <= 0:
            self.set_propagate_natural_width(True)
            self.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.NEVER)
            self.set_child(child)
            return

        speed = max(self.MIN_SPEED, min(speed, self.MAX_SPEED))

        self.set_min_content_width(width)
        self.set_size_request(width, -1)
        self.set_policy(Gtk.PolicyType.EXTERNAL, Gtk.PolicyType.NEVER)

        self.viewport = Gtk.Viewport.new()
        self.viewport.set_child(child)
        self.viewport.set_scroll_to_focus(False)

        h_adj = self.get_hadjustment()

        def _scroll(_ctrl, _dx, dy):
            h_adj.set_value(h_adj.get_value() + dy * speed)
            return True

        scroll_ctrl = Gtk.EventControllerScroll.new(
            Gtk.EventControllerScrollFlags.VERTICAL
        )
        scroll_ctrl.connect("scroll", guard(_scroll))
        self.viewport.add_controller(scroll_ctrl)
        self.set_child(self.viewport)


class BarWidget(Gtk.Box):
    """Base class of all bar widgets.

    Widgets are constructed on GTK main thread, their `run()` coroutine runs
    on a separate event loop thread. Methods ending with `_i` are safe to call
    from any thread, they schedule GTK updates on the main thread.
    """

    TYPE: str
    UNIQUE = True
    STATIC = False
    # spread wake-ups of widgets over time, see `ResourceManager.relax_interval()`
    RELAX_INTERVAL = True
    # wake up at wall clock multiples of the interval
    ALIGN_INTERVAL = False
    # updated only every interval, a warning is logged if it is not set
    POLLING = False
    # longest sleep of aligned widgets, see `sleep_interval()`
    MAX_ALIGNED_SLEEP = 1.0

    def __init__(
        self, name: str, res_mgr: ResourceManager, cfg: dict[str, Any] | None = None
    ):
        super().__init__()

        self.res_mgr = res_mgr
        # for use outside of GTK main thread
        self.widget_name = name
        self.set_name(name)
        self.add_css_class("bar-widget")

        # lets themes style widgets by type, e.g. `.bar-widget.i3_mode`
        if wtype := getattr(self, "TYPE", None):
            self.add_css_class(wtype)

        self.cfg = res_mgr.get_cfg_for_name(name) if cfg is None else cfg

        interval = self.cfg.get("interval", 0)

        if not isinstance(interval, (int, float)) or interval < 0:
            raise BarConfigError("'interval' must be a non-negative number")

        phase = 0.0

        if interval > 0:
            if self.RELAX_INTERVAL and self.cfg.get("relax_interval", True):
                interval, phase = res_mgr.relax_interval(interval)

            if self.ALIGN_INTERVAL:
                self._period = float(interval)
            else:
                self._period = next_prime(int(interval * 1000)) / 1000
        else:
            self._period = 0.0

            if self.POLLING:
                logging.warning(
                    "widget '%s' has no 'interval', it is never updated", name
                )

        self.interval = interval
        # delay of wake-ups after the first one, a fraction of the period
        self._phase = phase * self._period
        self._deadline: float | None = None

        # set by the bar while `run()` is running
        self.cancel_scope: anyio.CancelScope | None = None
        # event loop thread only
        self._available = True
        # GTK main thread only
        self._failed = False

        if (onclick := self.cfg.get("onclick")) is not None:
            if not isinstance(onclick, list):
                raise BarConfigError("'onclick' must be a list of commands")

            # the first command is for the primary (left) button
            for button, cmdline in enumerate(onclick, start=1):
                if cmdline:
                    self.onclick_exec(button, cmdline, f"'onclick' {button}")

        if (onscroll := self.cfg.get("onscroll")) is not None:
            if not isinstance(onscroll, list) or len(onscroll) != 2:
                raise BarConfigError("'onscroll' must be a list of two commands")
            self.onscroll_exec(*onscroll)

    # Lifecycle

    async def sleep_interval(self) -> bool:
        """Returns immediately when called for the first time, then sleeps for
        the configured interval. Returns `False` if the widget should stop."""

        if self._deadline is None:
            self._deadline = anyio.current_time() + self._phase
            return True

        if self._period <= 0:
            return False

        if self.ALIGN_INTERVAL:
            # wake up just after the boundary, never before it
            target = (time.time() // self._period + 1) * self._period + 0.005

            # the event loop clock stops while the system is suspended, the
            # wall clock does not: sleep in steps to catch up soon after resume
            while (remaining := target - time.time()) > 0:
                await anyio.sleep(min(remaining, self.MAX_ALIGNED_SLEEP))
        else:
            now = anyio.current_time()
            # do not try to catch up if the widget fell behind
            self._deadline = max(self._deadline + self._period, now)
            await anyio.sleep_until(self._deadline)

        return True

    async def run_wrapper(self):
        if not self.STATIC:
            await self.run()

    async def run(self):
        raise NotImplementedError()

    # Failures, GTK main thread

    def fail(self, ex: BaseException):
        """Disables the widget because of `ex`: removes it from the bar and
        stops `run()`."""
        if self._failed:
            return

        self._failed = True

        logging.error(
            "disabling widget '%s': %s",
            self.widget_name,
            describe_exception(ex),
            extra={"exception": ex},
        )

        self.teardown()

        if (parent := self.get_parent()) is not None:
            parent.remove(self)

        if (scope := self.cancel_scope) is not None:
            self.call_soon(scope.cancel)

    def teardown(self):
        """Called when the widget is disabled, e.g. to remove GLib sources."""

    def guard(self, func: Callable) -> Callable:
        """Wraps a GTK main thread callback, e.g. a signal handler: if it
        raises, the widget is disabled. Once the widget is disabled, the
        callback is not called and returns `None`, which also removes GLib
        timeout and idle sources."""

        @wraps(func)
        def _guarded(*args, **kwargs):
            if self._failed:
                return None
            try:
                return func(*args, **kwargs)
            except Exception as ex:
                self.fail(ex)
                return None

        return _guarded

    # Thread helpers

    def idle_add(self, func: Callable, *args: Any):
        """Calls `func` on GTK main thread."""
        guarded = self.guard(func)

        def _run():
            guarded(*args)
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

    def set_available_i(self, state: bool):
        """Hides the widget while its source is temporarily unavailable, e.g.
        a device is unplugged or a service is restarting. Event loop thread
        only."""
        if state != self._available:
            self._available = state
            self.set_visible_i(state)

    # Input

    def _onclick(self, button: int, action: ActionInterface):
        if button >= 0 and action is not None:
            controller = Gtk.GestureClick.new()
            controller.set_button(button)
            controller.connect("pressed", self.guard(lambda *_: action.run()))
            self.add_controller(controller)

    def _onscroll(
        self, action_up: ActionInterface | None, action_down: ActionInterface | None
    ):
        def _scroll(_ctrl, _dx: float, dy: float) -> bool:
            if dy < 0 and action_up is not None:
                action_up.run()
            elif dy > 0 and action_down is not None:
                action_down.run()
            return True

        if action_up is not None or action_down is not None:
            controller = Gtk.EventControllerScroll.new(
                Gtk.EventControllerScrollFlags.VERTICAL
                | Gtk.EventControllerScrollFlags.DISCRETE
            )
            controller.connect("scroll", self.guard(_scroll))
            self.add_controller(controller)

    def onclick_call(self, button: int, func: Callable, *args, **kwargs):
        """`func` is called on GTK main thread."""
        self._onclick(button, CallableAction(func, *args, **kwargs))

    def onclick_exec(
        self, button: int, cmdline: str | list[str], option: str = "command"
    ):
        self._onclick(button, ExecAction(cmdline, option))

    def onscroll_call(self, func_up: Callable, func_down: Callable):
        """Functions are called on GTK main thread."""
        self._onscroll(CallableAction(func_up), CallableAction(func_down))

    def onscroll_exec(self, cmdline_up: str | list[str], cmdline_down: str | list[str]):
        """An empty command does nothing."""
        self._onscroll(
            ExecAction(cmdline_up, "'onscroll' up") if cmdline_up else None,
            ExecAction(cmdline_down, "'onscroll' down") if cmdline_down else None,
        )


class WidgetBase(BarWidget):
    """A widget showing a label and, optionally, an icon. The icon is laid out
    according to the first icon specification found in the label, the ramp
    or `layout_specs`, texts the widget may show otherwise."""

    DFL_RAMP_IDX = 100
    # ramp entries are for states, e.g. off, on, connected: the ramp level is
    # the index of the entry
    RAMP_BY_STATE = False

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

        # parsed upfront to report invalid entries on startup
        self._ramp_contents = [WidgetContent.parse(r) for r in self.ramp]

        self.max_ramp_level = self.cfg.get("max_ramp_level", self.DFL_RAMP_IDX)

        # format strings of the labels before and after the icon
        self._label_fmt = self.content.label
        self._label_end_fmt: str | None = None
        self.label_end: Gtk.Label | None = None

        icon_position = self._icon_position(layout_specs)

        if icon_position == IconPosition.START:
            self.icon.add_css_class("bar-widget-icon-start")
            self.append(self.icon)
            self.append(self.label)
        elif icon_position == IconPosition.END:
            self.icon.add_css_class("bar-widget-icon-end")
            self.append(self.label)
            self.append(self.icon)
        elif icon_position == IconPosition.MIDDLE:
            self.label_end = Gtk.Label.new()
            self.label_end.add_css_class("bar-widget-label")
            self.icon.add_css_class("bar-widget-icon-middle")
            self.append(self.label)
            self.append(self.icon)
            self.append(self.label_end)
        else:
            self.append(self.label)

        labels = (
            [self.label] if self.label_end is None else [self.label, self.label_end]
        )

        # shown once there is an icon to show
        self.icon.set_visible(False)

        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.CENTER)
        self.set_homogeneous(False)

        self.icon.set_pixel_size(self.res_mgr.pixel_size)

        self.formatter = OptionalFormatter()

        width_chars = self.cfg.get("width_chars", 0)
        max_width_chars = self.cfg.get("max_width_chars", 0)

        for label_ in labels:
            label_.set_xalign(0.5)
            label_.set_yalign(0.5)
            label_.set_single_line_mode(True)

            if width_chars > 0:
                label_.set_width_chars(width_chars)

            if max_width_chars > 0:
                label_.set_max_width_chars(max_width_chars)
                label_.set_ellipsize(Pango.EllipsizeMode.END)

        self._updater = IdleUpdater(self.guard(self.apply_content))

        # State of GTK widgets, GTK main thread only
        self._shown_icon: str | None = None
        self._shown_label: str | None = None
        self._shown_label_end: str | None = None
        self._shown_tooltip: str | None = None
        self._shown_classes: dict[Gtk.Widget, frozenset[str]] = {
            self: frozenset(),
            self.icon: frozenset(),
        }
        self._shown_classes.update((label_, frozenset()) for label_ in labels)

    def _icon_position(self, layout_specs: Iterable[str]) -> IconPosition:
        """A `{ramp}` field places ramp icons, otherwise the first icon
        position found in the label, the ramp or `layout_specs` is used.
        Sets up label format strings."""

        ramp_has_icons = any(
            r.icon_position != IconPosition.NONE for r in self._ramp_contents
        )

        if (
            ramp_has_icons
            and (parts := split_format(self.content.label, "ramp")) is not None
        ):
            before, ramp_field, after = parts

            # the ramp label, if any, goes after the icon
            if not before.strip():
                return IconPosition.START
            if not after.strip():
                return IconPosition.END

            self._label_fmt = before
            self._label_end_fmt = ramp_field + after
            return IconPosition.MIDDLE

        if self.content.icon_position != IconPosition.NONE:
            return self.content.icon_position

        for content in (*self._ramp_contents, *map(WidgetContent.parse, layout_specs)):
            if content.icon_position != IconPosition.NONE:
                return content.icon_position

        return IconPosition.NONE

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

        if self.RAMP_BY_STATE:
            return min(ramp_level, len(self.ramp) - 1)

        return spread_index(ramp_level, self.max_ramp_level, len(self.ramp))

    def get_ramp(self, ramp_level: int = -1) -> WidgetContent | None:
        if (idx := self.ramp_index(ramp_level)) is None:
            return None
        return self._ramp_contents[idx]

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

        label = self.formatter.vformat(self._label_fmt, (), fields).strip()
        label_end = None

        if self._label_end_fmt is not None:
            label_end = self.formatter.vformat(self._label_end_fmt, (), fields).strip()

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
            label_end,
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

        icon = None if content.icon == NO_ICON else content.icon

        if icon != self._shown_icon:
            self._shown_icon = icon

            # e.g. ramp entries having only CSS classes
            if icon is None:
                self.icon.clear()
            else:
                self.icon.set_from_paintable(self.res_mgr.get_paintable(icon))

            self.icon.set_visible(icon is not None)

        if content.label is not None and content.label != self._shown_label:
            self._shown_label = content.label
            self.label.set_label(content.label)

        if self.label_end is not None and content.label_end != self._shown_label_end:
            self._shown_label_end = content.label_end
            self.label_end.set_label(content.label_end or "")

        if content.tooltip_text != self._shown_tooltip:
            self._shown_tooltip = content.tooltip_text
            self.set_tooltip_text(content.tooltip_text)

        self._sync_css_classes(self, content.widget_classes)
        self._sync_css_classes(self.label, content.label_classes)
        self._sync_css_classes(self.icon, content.icon_classes)

        if self.label_end is not None:
            self._sync_css_classes(self.label_end, content.label_classes)
