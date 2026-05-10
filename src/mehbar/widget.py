import json
import logging
import pickle
import re
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from enum import Enum
from functools import cache, partial
from pathlib import Path
from typing import Any, Sequence

import anyio
from gi.repository import Gdk, GdkPixbuf, GLib, Gtk
from i3ipc.aio import Connection

from mehbar.actions import (
    ActionInterface,
    CallableAction,
    ExecAction,
    GestureMouseClick,
)
from mehbar.exceptions import WidgetTerminated
from mehbar.tools import OptionalFormatter, md5sum_sync


class IconPosition(Enum):
    START = 0
    END = 1
    NONE = 2


# class Hero:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age

#     def __str__(self):
#         return self.name + str(self.age)

#     def __hash__(self):
#         print(hash(str(self)))
#         return hash(str(self))

#     def __eq__(self,other):
#         return self.name == other.name and self.age== other.age
# RE_SPEC = re.compile(
#     r"""\[\s*(?:
#     (?:icon\s*=\s*(?P<icon>\w+)\s*(?P<pos>[<>])?\s*(?:;)\s*)
#     |(?:classes\s*:\s*(?:(?:\s*(?:,)?\s*label\s*=\s*(?P<lcls>[\w\ ]+))?
#     |(?:\s*(?:,)?\s*icon\s*=\s*(?P<icls>[\w\ ]+))?
#     |(?:\s*(?:,)?\s*widget\s*=\s*(?P<wcls>[\w\ ]+))?){1,3}\s*(?:;)\s*)
#     |(?:\s*tooltip\s*=\s*(?P<tooltip>[\w\ ]+)?\s*(?:;)\s*))
#     {1,3}\s*\]""",
#     re.X,
# )


@dataclass(frozen=True)
class WidgetContent:
    RE_SPEC = re.compile(
        r"""\[\s*(?:
        (?:icon\s*=\s*(?P<icon>\w+)\s*(?P<pos>[<>])?\s*(?:;)\s*)
        |(?:classes\s*:\s*(?:(?:\s*(?:,)?\s*label\s*=\s*(?P<lcls>[\w\_\-\!\ ]+))?
        |(?:\s*(?:,)?\s*icon\s*=\s*(?P<icls>[\w\_\-\!\ ]+))?
        |(?:\s*(?:,)?\s*widget\s*=\s*(?P<wcls>[\w\_\-\!\ ]+))?){1,3}\s*(?:;)\s*)
        |(?:\s*tooltip\s*=\s*(?P<tooltip>[\w\_\-\!\ ]+)?\s*(?:;)\s*))
        {1,3}\s*\]""",
        re.X,
    )

    icon: str | None
    icon_position: IconPosition
    label_text: str | None
    tooltip_text: str | None
    icon_classes: set[str] | None
    label_classes: set[str] | None
    widget_classes: set[str] | None

    def derive(
        self,
        icon: str | None,
        icon_position: IconPosition | None,
        label_text: str | None,
        tooltip_text: str | None,
        icon_classes: set[str] | None,
        label_classes: set[str] | None,
        widget_classes: set[str] | None,
    ) -> WidgetContent:

        if icon is None and self.icon is not None:
            icon_ = self.icon
        else:
            icon_ = icon

        if icon_position is None and self.icon_position is not None:
            icon_position_ = self.icon_position
        else:
            icon_position_ = icon_position

        if label_text is None and self.label_text is not None:
            label_text_ = self.label_text
        else:
            label_text_ = label_text

        if tooltip_text is None and self.tooltip_text is not None:
            tooltip_text_ = self.tooltip_text
        else:
            tooltip_text_ = tooltip_text

        if icon_classes is None and self.icon_classes is not None:
            icon_classes_ = self.icon_classes
        else:
            icon_classes_ = icon_classes

        if label_classes is None and self.label_classes is not None:
            label_classes_ = self.label_classes
        else:
            label_classes_ = label_classes

        if widget_classes is None and self.widget_classes is not None:
            widget_classes_ = self.widget_classes
        else:
            widget_classes_ = widget_classes

        return self.__class__(
            icon_,
            icon_position_,
            label_text_,
            tooltip_text_,
            icon_classes_,
            label_classes_,
            widget_classes_,
        )

    @classmethod
    @cache
    def parse(cls, text: str) -> WidgetContent:  # noqa: F821
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
                widget=<CLASS 1>, ..., <CLASS N>
                    Space-separated list of CSS classes to be applied to the widget
                icon=<CLASS 1>, ..., <CLASS N>
                    Space-separated list of CSS classes to be applied to the icon
                label=<CLASS 1>, ..., <CLASS N>
                    Space-separated list of CSS classes to be applied to the label
            TOOLTIP
                tooltip=<TOOLTIP TEXT>
                    ...
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
        label_text = None

        if text is not None:
            if (match_spec := cls.RE_SPEC.search(text)) is not None:
                icon = match_spec.group("icon")

                match match_spec.group("pos"):
                    case ">":
                        icon_position = IconPosition.END
                    case "<":
                        icon_position = IconPosition.START

                if (icon_classes_ := match_spec.group("icls")) is not None:
                    icon_classes = set(icon_classes_.split())

                if (label_classes_ := match_spec.group("lcls")) is not None:
                    label_classes = set(label_classes_.split())

                if (widget_classes_ := match_spec.group("wcls")) is not None:
                    widget_classes = set(widget_classes_.split())

                tooltip_text = match_spec.group("tooltip")

            label_text = cls.RE_SPEC.sub("", text)

        return cls(
            icon,
            icon_position,
            label_text,
            tooltip_text,
            icon_classes,
            label_classes,
            widget_classes,
        )


class IconManager:
    MIN_SIZE = 8

    def __init__(self, pixel_size: int):
        self.pixel_size = max(self.MIN_SIZE, pixel_size)
        self.store = {}
        self.cksums = {}

        self.theme = Gtk.IconTheme.get_for_display(Gdk.Display.get_default())

    def load_image(self, name: str, path: str | Path):

        if isinstance(path, str) and path.startswith("theme:"):
            paintable = self.theme.lookup_icon(
                path.lstrip("theme:").strip(),
                None,
                self.pixel_size,
                1,
                Gtk.TextDirection.NONE,
                Gtk.IconLookupFlags.NONE,
            )

            if paintable:
                self.store[name] = paintable.get_current_image()
        else:
            cksum = md5sum_sync(path)

            if cksum in self.cksums and self.cksums[cksum] in self.store:
                self.store[name] = self.store[self.cksums[cksum]]
            else:
                pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                    path, -1, self.pixel_size, preserve_aspect_ratio=True
                )
                self.store[name] = Gdk.Texture.new_for_pixbuf(
                    pixbuf
                ).get_current_image()
                self.cksums[cksum] = name

    def get_texture(self, name: str) -> Gdk.Paintable:
        return self.store[name]


class RewriteMixin:
    def __init__(self, *args, rewrite: dict[str, str] | None, **kwargs):
        super().__init__(*args, **kwargs)
        self._rewrite = rewrite

    def rewrite(self, text: str) -> str:
        result = text
        if self._rewrite is not None and text is not None:
            for pattern, repl in self._rewrite.items():
                if re.match(pattern, text) is not None:
                    result = re.sub(pattern, repl, text)
                    break
        return result


class I3ListenerMixin:
    def __init__(self, *args, i3_conn: Connection, **kwargs):
        super().__init__(*args, **kwargs)
        self._i3_conn = i3_conn

    async def get_i3_conn(self):
        if self._i3_conn is None:
            self._i3_conn = await Connection().connect()
        return self._i3_conn


class JSONInputMixin:
    MAX_RAMP = 100
    MAX_LPS = 10

    def __init__(self, *args, max_lps: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_lps = max(min(max_lps, self.MAX_LPS), 1)

    async def format_label_idle_json_async(self, json_str: str):
        if json_str.strip():
            try:
                val_map = json.loads(json_str)
                if "ramp" not in val_map:
                    ramp_ = None

                    if self.ramps and (level := val_map.get("ramp_level")) is not None:
                        try:
                            ramp_ = self.ramps[min(max(int(level), 0), self.MAX_RAMP)]
                        except ValueError as ex:
                            logging.error("invalid value in 'ramp_level' field: %s", ex)

                    val_map["ramp"] = ramp_

                self.format_label_idle(**val_map)
            except json.JSONDecodeError as ex:
                logging.error("failed to parse JSON input: %s", ex)
        await anyio.sleep(0.1)


class WidgetBase(Gtk.Box):
    UNIQUE = True

    def __init__(
        self,
        interval: int = 0,
        label_format: str | None = None,
        ramp: list[str] | None = None,
        icon_manager: IconManager | None = None,
        max_ramp_level: int = 100,
    ):
        super().__init__()

        self.label = Gtk.Label.new()
        self.label.add_css_class("bar-widget-label")

        self.icon = Gtk.Image.new()
        self.icon.add_css_class("bar-widget-icon")

        self.init_content = WidgetContent.parse(label_format)

        if self.init_content.icon_position == IconPosition.START:
            self.append(self.icon)
            self.append(self.label)
        elif self.init_content.icon_position == IconPosition.END:
            self.append(self.label)
            self.append(self.icon)
        else:
            self.append(self.label)

        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.CENTER)
        self.set_homogeneous(False)

        self.icon_manager = icon_manager

        if self.icon_manager is not None:
            self.icon.set_pixel_size(self.icon_manager.pixel_size)

        self.set_label = self.label.set_label
        self.set_from_paintable = self.icon.set_from_paintable

        self._init_loop = False
        self.loop_token = None

        self._last_content = None
        self._last_icon = None
        self._last_label_text = None
        self._last_css_classes = set()
        self._last_label_css_classes = set()
        self._last_icon_css_classes = set()

        self.cache: dict[str, Any] = {}
        self.formatter = OptionalFormatter()
        self.interval = max(int(interval), 0)

        self.ramp = ramp
        self.max_ramp_level = max_ramp_level
        self.ramp_index_cache = {}

        self.content_cache = {}

        self.label.set_xalign(0.5)
        self.label.set_yalign(0.5)
        self.label.set_single_line_mode(True)

        self.add_css_class("bar-widget")

        self.set_icon(self.init_content.icon)

    async def sleep_interval(self) -> bool:

        if self._init_loop:
            if self.interval > 0:
                self._init_loop = True
                await anyio.sleep(self.interval)
            else:
                return False
        else:
            self._init_loop = True
        return True

    def shutdown(self):
        self.interval = -1

    def stop(self):
        raise WidgetTerminated()

    async def run_wrapper(self):
        self.loop_token = anyio.lowlevel.current_token()
        await self.run()

    def _idle_run(self, func: Callable, *args: Any) -> bool:
        func(*args)
        return GLib.SOURCE_REMOVE

    def idle_add(self, func: Callable, *args: Any):
        GLib.idle_add(self._idle_run, func, *args)

    def _idle_run_cb(self, func: Callable, cb: Callable) -> bool:
        func()
        if cb is not None:
            self.elt_run_sync(cb)
        return GLib.SOURCE_REMOVE

    def idle_add_cb(self, func: Callable, cb: Callable):
        GLib.idle_add(self._idle_run_cb, func, cb)

    def _add_css_classes(self, widget: Gtk.Widget, classes: set[str]):
        for class_name in classes:
            widget.add_css_class(class_name)

    def _add_css_classes_idle(
        self,
        widget: Gtk.Widget,
        classes: set[str],
        store: set[str],
    ):
        if classes and "!none" not in classes and not classes.issubset(store):
            self.idle_add_cb(
                partial(self._add_css_classes, widget, classes),
                partial(store.update, classes),
            )

    def _rm_css_classes(self, widget: Gtk.Widget, classes: set[str]):
        for class_name in classes:
            widget.remove_css_class(class_name)

    def _rm_css_classes_idle(
        self,
        widget: Gtk.Widget,
        classes: set[str],
        store: set[str],
    ):
        if not classes or "!none" in classes:
            if store:
                self.idle_add_cb(
                    partial(self._rm_css_classes, widget, store), store.clear
                )
        else:
            self.idle_add_cb(
                partial(self._rm_css_classes, widget, classes),
                partial(store.difference_update, classes),
            )

    def add_icon_css_classes_idle(self, classes: set[str]):
        self._add_css_classes_idle(self.icon, classes, self._last_icon_css_classes)

    def remove_icon_css_classes_idle(self, classes: set[str] | None = None):
        self._rm_css_classes_idle(self.icon, classes, self._last_icon_css_classes)

    def replace_icon_css_classes_idle(self, classes: set[str]):
        if classes and not classes.issubset(self._last_icon_css_classes):
            self.remove_icon_css_classes_idle(classes)
            self.add_icon_css_classes_idle(classes)

    def add_label_css_classes_idle(self, classes: Sequence[str]):
        pass

    def remove_label_css_classes_idle(self, classes: Sequence[str] | None = None):
        pass

    def replace_label_css_classes_idle(self, classes: Sequence[str]):
        pass

    def add_css_classes_idle(self, classes: Sequence[str]):
        pass

    def remove_css_classes_idle(self, classes: set[str]):
        pass

    def replace_css_classes_idle(self, classes: set[str]):
        pass

    def set_content_idle(self, content: WidgetContent, **kwargs):
        if content != self._last_content:
            if content.icon is not None:
                self.set_icon_idle(content.icon)

            if content.label_text is not None:
                self.set_label_idle(content.label_text)

            if content.widget_classes:
                self.replace_css_classes_idle(content.widget_classes)

            if content.label_classes:
                self.replace_label_css_classes_idle(content.label_classes)

            if content.icon_classes:
                self.replace_icon_css_classes_idle(content.icon_classes)

    def set_icon(self, name: str):
        if name != self._last_icon:
            self._last_icon = name
            self.icon.set_from_paintable(self.icon_manager.get_texture(name))

    def set_icon_idle(self, name: str):
        self.idle_add(self.set_icon, name)

    def _get_ramp(self, ramp_level: int = -1) -> WidgetContent | None:

        if ramp_level not in self.ramp_index_cache:
            content = None

            if ramp_level >= 0 and self.ramp is not None:
                idx = int(
                    min(ramp_level, self.max_ramp_level - 1)
                    / (self.max_ramp_level / len(self.ramp))
                )
                content = WidgetContent.parse(self.ramp[idx])

            self.ramp_index_cache[ramp_level] = content

        return self.ramp_index_cache[ramp_level]

    def get_content(self, ramp_level: int = -1, **kwargs: str) -> WidgetContent:

        key = pickle.dumps(kwargs, protocol=pickle.HIGHEST_PROTOCOL)

        if key not in self.content_cache:
            icon = None
            label_text = None
            tooltip_text = None
            icon_classes = None
            label_classes = None
            widget_classes = None

            if (ramp_content := self._get_ramp(ramp_level)) is not None:
                icon = ramp_content.icon
                icon_classes = ramp_content.icon_classes

                if self.init_content.label_text is not None:
                    label_text = self.formatter.vformat(
                        self.init_content.label_text,
                        None,
                        {**kwargs, "ramp": ramp_content.label_text},
                    )

                if self.init_content.tooltip_text is not None:
                    tooltip_text = self.formatter.vformat(
                        self.init_content.tooltip_text,
                        None,
                        {**kwargs, "ramp": ramp_content.tooltip_text},
                    )

                icon_classes = ramp_content.icon_classes
                label_classes = ramp_content.label_classes
                widget_classes = ramp_content.widget_classes
            else:
                if self.init_content.label_text is not None:
                    label_text = self.formatter.vformat(
                        self.init_content.label_text,
                        None,
                        kwargs,
                    )
                if self.init_content.tooltip_text is not None:
                    tooltip_text = self.formatter.vformat(
                        self.init_content.tooltip_text,
                        None,
                        kwargs,
                    )

            logging.debug("GETTING CONTENT")

            self.content_cache[key] = self.init_content.derive(
                icon,
                None,
                label_text,
                tooltip_text,
                icon_classes,
                label_classes,
                widget_classes,
            )
        return self.content_cache[key]

    def _onclick(self, button: int, action: ActionInterface):
        if button >= 0 and action is not None:
            controller = GestureMouseClick()
            controller.set_button(button)
            controller.connect("pressed", lambda *args: action.run())
            self.add_controller(controller)

    def _onscroll(self, action_up: ActionInterface, action_down: ActionInterface):

        def _scroll(x: float, dx: float, dy: float):
            if dy > 0:
                action_up.run()
            else:
                action_down.run()

        if action_up is not None and action_down is not None:
            controller = Gtk.EventControllerScroll.new(
                Gtk.EventControllerScrollFlags.VERTICAL
            )
            controller.connect("scroll", _scroll)
            self.add_controller(controller)

    def vsformat(self, **kwargs):
        return self.formatter.format(self.init_content.label_text, **kwargs)

    def set_visible_idle(self, state: bool):
        self.idle_add(self.set_visible, state)

    def set_label_idle(self, text: str):
        if text != self._last_label_text:
            self._last_label_text = text
            self.idle_add(self.set_label, text)

    def format_label_idle(self, **kwargs):
        self.set_label_idle(self.vsformat(**kwargs).strip())
        # TODO: remove this

    def onclick_call(self, button: int, func: Callable, *args, **kwargs):
        self._onclick(button, CallableAction(func, *args, **kwargs))

    def onclick_exec(self, button: int, cmdline: str | list[str]):
        self._onclick(button, ExecAction(cmdline))

    def onscroll_call(self, func_up: Callable, func_down: Callable):
        self._onscroll(CallableAction(func_up), CallableAction(func_down))

    def onscroll_exec(self, cmdline_up: str | list[str], cmdline_down: str | list[str]):
        self._onscroll(ExecAction(cmdline_up), ExecAction(cmdline_down))

    def elt_run_sync(self, func: Callable, *args):
        anyio.from_thread.run_sync(func, *args, token=self.loop_token)

    def elt_run(self, coro: Coroutine, *args):
        anyio.from_thread.run(coro, *args, token=self.loop_token)


class JSONInputWidgetBase(JSONInputMixin, WidgetBase):
    def __init__(
        self,
        interval: int,
        label_format: str,
        ramp: list[str] | None = None,
        max_lps: int = 0,
    ):
        super().__init__(interval, label_format, ramp, max_lps=max_lps)

        self.ramps = []

        if ramp is not None:
            for level in range(self.MAX_RAMP + 1):
                ramp_val = str()

                if ramp and (nramp := len(ramp)) > 0:
                    ramp_idx = int(
                        min(level, self.MAX_RAMP - 1) / (self.MAX_RAMP / nramp)
                    )
                    ramp_val = ramp[ramp_idx]

                    self.ramps.append(ramp_val)
