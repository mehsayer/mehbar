from i3ipc import Event, ModeEvent

from mehbar.resource_manager import ResourceManager
from mehbar.widget import RewriteMixin, WidgetBase

from ._i3 import I3Connection, I3WidgetMixin


class WidgetI3Mode(I3WidgetMixin, RewriteMixin, WidgetBase):
    """Shows the binding mode. Unless `always_show` is set, the widget is
    hidden in the default mode."""

    TYPE = "i3_mode"
    DEFAULT_MODE = "default"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.always_show = self.cfg.get("always_show", True)

    def _show_mode(self, mode: str):
        self.set_new_content_i(mode=self.rewrite(mode))
        self.set_visible_i(self.always_show or mode != self.DEFAULT_MODE)

    async def i3_start(self, conn: I3Connection):
        def _callback_mode(event: ModeEvent):
            self._show_mode(event.change)

        self.i3_on(conn, Event.MODE, _callback_mode)

        # i3ipc.aio cannot query the current binding state
        self._show_mode(self.DEFAULT_MODE)
