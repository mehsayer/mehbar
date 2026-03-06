from mehbar.widgets import BarWidget, RewriteMixin, I3ListenerMixin
from i3ipc.aio import Connection
from i3ipc import Event

class BarWidgetI3Scratchpad(I3ListenerMixin, BarWidget):
    def __init__(self, label_format: str, always_show: bool, i3_conn: Connection):
        super().__init__(0, label_format, i3_conn=i3_conn)
        self.always_show = always_show

    async def run(self):

        conn = await self.get_i3_conn()

        def _dispatch_scratchpad(con: Con):
            num = 0

            if con is not None:
                if (scrpad := con.scratchpad()) is not None:
                    num = len(scrpad.nodes) + len(scrpad.floating_nodes)
                    if self._last_value != num:
                        self._last_value = num
                        self.format_label_idle(count=num)

            self.set_visible(not (self.always_show and num == 0))

        async def _callback_scratchpad(*_):
            _dispatch_scratchpad(await conn.get_tree())

        _dispatch_scratchpad(await conn.get_tree())

        conn.on(Event.WINDOW_MOVE, _callback_scratchpad)
