import anyio

from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


class WidgetI3Scratchpad(WidgetBase):
    """Shows the number of windows in the scratchpad as the `count` field.
    Unless `always_show` is set, the widget is hidden when there are none."""

    TYPE = "i3_scratchpad"

    # window events that may change the scratchpad
    CHANGES = frozenset({"move", "close", "floating"})

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.always_show = self.cfg.get("always_show", True)

    async def run(self):
        from i3ipc import Event, WindowEvent

        i3_conn = await self.res_mgr.get_i3_connection_async()

        async def _show_count():
            num = 0

            if (scrpad := (await i3_conn.get_tree()).scratchpad()) is not None:
                num = len(scrpad.nodes) + len(scrpad.floating_nodes)

            self.set_new_content_i(num, count=num)
            self.set_visible_i(self.always_show or num > 0)

        async def _callback_window(_, event: WindowEvent):
            if event.change in self.CHANGES:
                await _show_count()

        await _show_count()

        i3_conn.on(Event.WINDOW, _callback_window)

        try:
            await anyio.sleep_forever()
        finally:
            i3_conn.off(_callback_window)
