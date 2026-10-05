import anyio

from mehbar.resource_manager import ResourceManager
from mehbar.widget import RewriteMixin, WidgetBase


class WidgetI3Mode(RewriteMixin, WidgetBase):
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

    async def run(self):
        from i3ipc import Event, ModeEvent

        i3_conn = await self.res_mgr.get_i3_connection_async()

        # i3ipc.aio cannot query the current binding state
        self._show_mode(self.DEFAULT_MODE)

        def _callback_mode(_, event: ModeEvent):
            self._show_mode(event.change)

        i3_conn.on(Event.MODE, _callback_mode)

        try:
            await anyio.sleep_forever()
        finally:
            i3_conn.off(_callback_mode)
