from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


class WidgetI3Scratchpad(WidgetBase):
    TYPE = "i3_scratchpad"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.always_show = self.cfg.get("always_show", False)
        self._last_num = -1

    async def run(self):

        from i3ipc import Event
        from i3ipc.aio import Con

        i3_conn = await self.res_mgr.get_i3_connection_async()

        def _dispatch_scratchpad(con: Con):
            num = 0

            if con is not None and (scrpad := con.scratchpad()) is not None:
                num = len(scrpad.nodes) + len(scrpad.floating_nodes)
                if self._last_num != num:
                    self._last_num = num
                    self.set_new_content_i(count=str(num))

            self.set_visible_idle(not (self.always_show and num == 0))

        async def _callback_scratchpad(*_) -> None:
            _dispatch_scratchpad(await i3_conn.get_tree())

        _dispatch_scratchpad(await i3_conn.get_tree())

        i3_conn.on(Event.WINDOW_MOVE, _callback_scratchpad)
