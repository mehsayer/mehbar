import re
import asyncio

from collections.abc import Callable
from typing import Any

from gi.repository import Gtk, GLib
from i3ipc.aio import Connection

from mehbar.tools import OptionalFormatter
from mehbar.actions import Action, CallableAction


class GestureMouseClick(Gtk.GestureClick, Gtk.GestureSingle):
    pass


class BarWindgetInterface:

    async def run(self):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()

    def vformat_label(self, **kwargs):
        raise NotImplementedError()

    def set_label_idle(self, label: str):
        raise NotImplementedError()

    def format_label_idle(self, **kwargs):
        raise NotImplementedError()

    def onclick_call(self, button: int, func: Callable, *args, **kwargs):
        raise NotImplementedError()

    def onscroll_call(self, func_up: Callable, func_down: Callable):
        raise NotImplementedError()

    def stop(self):
        pass


class BarWidget(BarWindgetInterface, Gtk.Label):

    def __init__(
        self,
        interval: int = 0,
        label_format: str | None = None,
        ramp: list[str] | None = None,
    ):
        super().__init__()

        self._run = True
        self._last_value: Any | None = None
        self._last_text: str | None = None
        self.cache: dict[str, Any] = {}
        self.formatter = OptionalFormatter()
        self.interval = max(int(interval), 0)
        self.label_format = label_format if label_format is not None else ""
        self.ramp = ramp

        self.set_xalign(0.5)
        self.set_yalign(0.5)
        self.set_single_line_mode(True)

        self.add_css_class("bar-widget")

    async def run(self):
        self.update()

        if self.interval > 0:
            while self._run:
                self.update()
                await asyncio.sleep(self.interval)

    def _set_label_idle(self, label: str) -> bool:
        """Calls Widget.set_label and returns False, so that it can be removed
        from event sources.
        """
        super().set_label(label.strip())
        return GLib.SOURCE_REMOVE

    def _onclick(self, button: int, action: Action):
        controller = GestureMouseClick()
        controller.set_button(button)
        controller.connect("pressed", lambda *args: action.run())
        self.add_controller(controller)

    def _onscroll(self, action_up: Action, action_down: Action):

        def _scroll(x: float, dx: float, dy: float):
            if dy > 0:
                action_up.run()
            else:
                action_down.run()

        controller = Gtk.EventControllerScroll.new(
            Gtk.EventControllerScrollFlags.VERTICAL
        )
        controller.connect("scroll", _scroll)
        self.add_controller(controller)


    # TODO: rename to vsformat
    def vformat_label(self, **kwargs):
        return self.formatter.format(self.label_format, **kwargs)

    def set_label_idle(self, label: str):
        if self._last_text != label:
            self._last_text = label
            GLib.idle_add(self._set_label_idle, label)

    def format_label_idle(self, **kwargs):
        self.set_label_idle(self.vformat_label(**kwargs).strip())

    def onclick_call(self, button: int, func: Callable, *args, **kwargs):
        self._onclick(button, CallableAction(func, *args, **kwargs))

    def onscroll_call(self, func_up: Callable, func_down: Callable):
        self._onscroll(CallableAction(func_up), CallableAction(func_down))

    def update(self):
        raise NotImplementedError()

    def stop(self):
        self._run = False


class RewriteMixin:
    def __init__(self, *args, rewrite: dict[str, str], **kwargs):
        super().__init__(*args, **kwargs)
        self._rewrite = rewrite

    def rewrite(self, text: str) -> str:
        result = text
        if text is not None:
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
