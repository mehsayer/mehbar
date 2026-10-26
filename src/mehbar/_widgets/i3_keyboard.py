from i3ipc import Event, InputEvent

from mehbar.widget import RewriteMixin, WidgetBase

from ._i3 import I3Connection, I3WidgetMixin


class WidgetI3KeyboardLayout(I3WidgetMixin, RewriteMixin, WidgetBase):
    TYPE = "i3_kblayout"

    async def i3_start(self, conn: I3Connection):
        def _callback_kb_layout(event: InputEvent):
            if event.input.type == "keyboard" and event.input.xkb_active_layout_name:
                self._show_layout(event.input.xkb_active_layout_name)

        self.i3_on(conn, Event.INPUT, _callback_kb_layout)

        for i3_input in await conn.get_inputs():
            if i3_input.type == "keyboard" and i3_input.xkb_active_layout_name:
                self._show_layout(i3_input.xkb_active_layout_name)
                break

    def _show_layout(self, raw_layout: str):
        self.set_new_content_i(layout=self.rewrite(raw_layout))
