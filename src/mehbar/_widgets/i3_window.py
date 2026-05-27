from mehbar.resource_manager import ResourceManager
from mehbar.widget import RewriteMixin, WidgetBase


class WidgetI3Window(RewriteMixin, WidgetBase):
    TYPE = "i3_window"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.always_show = self.cfg.get("always_show", True)
        self._last_name = None
        self._name_cache = {}

    async def run(self):

        from i3ipc import Con, Event, WindowEvent

        if not self.always_show:
            self.set_visible_idle(False)

        i3_conn = await self.res_mgr.get_i3_connection_async()

        def _dispatch_con(con: Con):

            if con is not None and con:
                win_name = None
                if con.name is not None:
                    win_name = con.name
                elif con.app_id is not None:
                    win_name = con.app_id

                if win_name is not None:
                    self.set_visible_idle(True)

                    if self._last_name != win_name:
                        self._last_name = win_name
                        if win_name not in self._name_cache:
                            self._name_cache[win_name] = self.rewrite(win_name)
                        self.format_label_idle(title=self._name_cache[win_name])
                else:
                    if not self.always_show:
                        self.set_visible_idle(False)
            else:
                if not self.always_show:
                    self.set_visible_idle(False)

        # Find the focused window title, if any, on start
        tree = await i3_conn.get_tree()
        _dispatch_con(tree.find_focused())

        async def _callback_window(_, event: WindowEvent):
            if event.change == "focus":
                _dispatch_con(event.container)
            elif event.change == "close":
                # Find the focused window title after (possibly the last open)
                # window is closed
                tree = await i3_conn.get_tree()
                _dispatch_con(tree.find_focused())

        i3_conn.on(Event.WINDOW_FOCUS, _callback_window)
        i3_conn.on(Event.WINDOW_CLOSE, _callback_window)
