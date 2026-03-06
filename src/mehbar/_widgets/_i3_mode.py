from mehbar.widgets import BarWidget, RewriteMixin, I3ListenerMixin
from i3ipc.aio import Connection
from i3ipc import Event

class BarWidgetI3Mode(I3ListenerMixin, RewriteMixin, BarWidget):

    def __init__(self,
                 rewrite: dict[str, str],
                 label_format: str,
                 always_show: bool,
                 i3_conn: Connection):
        super().__init__(0,
                         label_format,
                         None,
                         rewrite=rewrite,
                         i3_conn=i3_conn)
        self.always_show = always_show

    async def run(self):

        def _dispatch_mode(cur_mode: str):
            if self._last_value != cur_mode:
                self._last_value = cur_mode

                if cur_mode not in self.cache:
                    mode = self.rewrite(cur_mode)
                    self.cache[cur_mode] = mode
                self.format_label_idle(mode=self.cache[cur_mode])

            self.set_visible(
                not (self.always_show and cur_mode == "default")
            )

        _dispatch_mode("default")

        async def _callback_mode(_: Connection, event: ModeEvent):
            _dispatch_mode(event.change)

        (await self.get_i3_conn()).on(Event.MODE, _callback_mode)
