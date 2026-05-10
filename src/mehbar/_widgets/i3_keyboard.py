from i3ipc import Event, InputEvent
from i3ipc.aio import Connection

from mehbar.widget import I3ListenerMixin, RewriteMixin, WidgetBase


class WidgetI3KeyboardLayout(I3ListenerMixin, RewriteMixin, WidgetBase):
    TYPE = "i3_kblayout"

    def __init__(
        self,
        label_format: str,
        i3_conn: Connection,
        rewrite: dict[str, str] | None = None,
        icon_manager=None,
    ):
        super().__init__(
            0,
            label_format,
            None,
            icon_manager=icon_manager,
            rewrite=rewrite,
            i3_conn=i3_conn,
        )

    async def run(self):
        for i3_i in await (await self.get_i3_conn()).get_inputs():
            if i3_i.xkb_active_layout_name is not None:
                self._push_layout(i3_i.xkb_active_layout_name)
                break

        def _callback_kb_layout(_, event: InputEvent):
            evinput = event.input
            if evinput.type == "keyboard":
                if self._last_value != evinput.xkb_active_layout_name:
                    self._last_value = evinput.xkb_active_layout_name
                    self._push_layout(evinput.xkb_active_layout_name)

        (await self.get_i3_conn()).on(Event.INPUT, _callback_kb_layout)

    def _push_layout(self, raw_layout: str):
        if raw_layout not in self.cache:
            layout = self.rewrite(raw_layout)
            self.cache[raw_layout] = layout
        self.format_label_idle(layout=self.cache[raw_layout])
