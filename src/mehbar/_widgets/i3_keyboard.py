from mehbar.resource_manager import ResourceManager
from mehbar.widget import RewriteMixin, WidgetBase


class WidgetI3KeyboardLayout(RewriteMixin, WidgetBase):
    TYPE = "i3_kblayout"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self._layout_cache = {}
        self._last_layout_name = None

    async def run(self):

        from i3ipc import Event, InputEvent

        i3_conn = await self.res_mgr.get_i3_connection_async()
        for i3_i in await i3_conn.get_inputs():
            if i3_i.xkb_active_layout_name is not None:
                self._push_layout(i3_i.xkb_active_layout_name)
                break

        def _callback_kb_layout(_, event: InputEvent):
            evinput = event.input
            if evinput.type == "keyboard":
                if self._last_layout_name != evinput.xkb_active_layout_name:
                    self._last_layout_name = evinput.xkb_active_layout_name
                    self._push_layout(evinput.xkb_active_layout_name)

        i3_conn.on(Event.INPUT, _callback_kb_layout)

    def _push_layout(self, raw_layout: str):
        if raw_layout not in self._layout_cache:
            self._layout_cache[raw_layout] = self.rewrite(raw_layout)
        self.set_new_content_i(layout=self._layout_cache[raw_layout])
