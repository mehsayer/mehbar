from mehbar.resource_manager import ResourceManager
from mehbar.widget import RewriteMixin, WidgetBase


class WidgetI3Mode(RewriteMixin, WidgetBase):
    TYPE = "i3_mode"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.always_show = self.cfg.get("always_show", False)
        self._mode_cache = {}
        self._last_mode = None

    async def run(self):

        from i3ipc import Event, ModeEvent

        i3_conn = await self.res_mgr.get_i3_connection_async()

        def _dispatch_mode(cur_mode: str):
            if self._last_mode != cur_mode:
                self._last_mode = cur_mode

                if cur_mode not in self._mode_cache:
                    mode = self.rewrite(cur_mode)
                    self._mode_cache[cur_mode] = mode
                self.set_new_content_i(mode=self._mode_cache[cur_mode])

            self.set_visible_idle(not (self.always_show and cur_mode == "default"))

        _dispatch_mode("default")

        def _callback_mode(_, event: ModeEvent) -> None:
            _dispatch_mode(event.change)

        i3_conn.on(Event.MODE, _callback_mode)
