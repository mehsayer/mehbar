from i3ipc import Event, WindowEvent

from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase

from ._i3 import I3Connection, I3WidgetMixin


class WidgetI3Scratchpad(I3WidgetMixin, WidgetBase):
    """Shows the number of windows in the scratchpad as the `count` field.
    Unless `always_show` is set, the widget is hidden when there are none."""

    TYPE = "i3_scratchpad"

    # window events that may change the scratchpad
    CHANGES = frozenset({"move", "close", "floating"})

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.always_show = self.cfg.get("always_show", True)

    async def i3_start(self, conn: I3Connection):
        async def _show_count():
            num = 0

            if (scrpad := (await conn.get_tree()).scratchpad()) is not None:
                num = len(scrpad.nodes) + len(scrpad.floating_nodes)

            self.set_new_content_i(num, count=num)
            self.set_visible_i(self.always_show or num > 0)

        async def _callback_window(event: WindowEvent):
            if event.change in self.CHANGES:
                await _show_count()

        self.i3_on(conn, Event.WINDOW, _callback_window)

        await _show_count()
