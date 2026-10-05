import anyio

from mehbar.widget import RewriteMixin, WidgetBase


class WidgetI3KeyboardLayout(RewriteMixin, WidgetBase):
    TYPE = "i3_kblayout"

    async def run(self):
        from i3ipc import Event, InputEvent

        i3_conn = await self.res_mgr.get_i3_connection_async()

        for i3_input in await i3_conn.get_inputs():
            if i3_input.type == "keyboard" and i3_input.xkb_active_layout_name:
                self._show_layout(i3_input.xkb_active_layout_name)
                break

        def _callback_kb_layout(_, event: InputEvent):
            if event.input.type == "keyboard" and event.input.xkb_active_layout_name:
                self._show_layout(event.input.xkb_active_layout_name)

        i3_conn.on(Event.INPUT, _callback_kb_layout)

        try:
            await anyio.sleep_forever()
        finally:
            i3_conn.off(_callback_kb_layout)

    def _show_layout(self, raw_layout: str):
        self.set_new_content_i(layout=self.rewrite(raw_layout))
