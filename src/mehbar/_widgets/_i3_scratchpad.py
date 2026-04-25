from i3ipc import Event
from i3ipc.aio import Connection

from mehbar.widgets import I3ListenerMixin, RewriteMixin, Widget


class WidgetI3Scratchpad(I3ListenerMixin, Widget):
    def __init__(self, label_format: str, always_show: bool, i3_conn: Connection):
        super().__init__(0, label_format, i3_conn=i3_conn)
        self.always_show = always_show

    async def run(self):

        conn = await self.get_i3_conn()

        def _dispatch_scratchpad(con: Con):
            num = 0

            if con is not None and (scrpad := con.scratchpad()) is not None:
                num = len(scrpad.nodes) + len(scrpad.floating_nodes)
                if self._last_value != num:
                    self._last_value = num
                    self.format_label_idle(count=num)

            self.set_visible_idle(not (self.always_show and num == 0))

        async def _callback_scratchpad(*_):
            _dispatch_scratchpad(await conn.get_tree())

        _dispatch_scratchpad(await conn.get_tree())

        conn.on(Event.WINDOW_MOVE, _callback_scratchpad)
