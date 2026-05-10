from i3ipc import Con, Event, WindowEvent
from i3ipc.aio import Connection

from mehbar.widget import I3ListenerMixin, RewriteMixin, WidgetBase


class WidgetI3Window(I3ListenerMixin, RewriteMixin, WidgetBase):
    TYPE = "i3_window"

    def __init__(
        self,
        label_format: str,
        i3_conn: Connection,
        rewrite: dict[str, str] | None = None,
        always_show: bool = True,
    ):
        super().__init__(0, label_format, None, rewrite=rewrite, i3_conn=i3_conn)
        self.always_show = always_show

    async def run(self):
        if not self.always_show:
            self.set_visible_idle(False)

        conn = await self.get_i3_conn()

        def _dispatch_con(con: Con):

            if con is not None and con:
                win_name = None
                if con.name is not None:
                    win_name = con.name
                elif con.app_id is not None:
                    win_name = con.app_id

                if win_name is not None:
                    self.set_visible_idle(True)

                    if self._last_value != win_name:
                        self._last_value = win_name
                        if win_name not in self.cache:
                            self.cache[win_name] = self.rewrite(win_name)
                        self.format_label_idle(title=self.cache[win_name])
                else:
                    if not self.always_show:
                        self.set_visible_idle(False)
            else:
                if not self.always_show:
                    self.set_visible_idle(False)

        # Find the focused window title, if any, on start
        tree = await conn.get_tree()
        _dispatch_con(tree.find_focused())

        async def _callback_window(_: Connection, event: WindowEvent):
            if event.change == "focus":
                _dispatch_con(event.container)
            elif event.change == "close":
                # Find the focused window title after (possibly the last open)
                # window is closed
                tree = await conn.get_tree()
                _dispatch_con(tree.find_focused())

        conn.on(Event.WINDOW_FOCUS, _callback_window)
        conn.on(Event.WINDOW_CLOSE, _callback_window)
