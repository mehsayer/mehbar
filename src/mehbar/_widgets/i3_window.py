import anyio

from mehbar.resource_manager import ResourceManager
from mehbar.widget import RewriteMixin, WidgetBase


class WidgetI3Window(RewriteMixin, WidgetBase):
    """Shows the title of the focused window as the `title` field. Unless
    `always_show` is set, the widget is hidden when no window is focused."""

    TYPE = "i3_window"

    WINDOW_TYPES = frozenset({"con", "floating_con"})

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.always_show = self.cfg.get("always_show", True)

    def _show_window(self, con):
        title = None

        if con is not None and con.type in self.WINDOW_TYPES:
            title = con.name or con.app_id or con.window_class

        self.set_new_content_i(title=self.rewrite(title) if title else "")
        self.set_visible_i(self.always_show or title is not None)

    async def run(self):
        from i3ipc import Event, WindowEvent

        i3_conn = await self.res_mgr.get_i3_connection_async()

        async def _show_focused():
            self._show_window((await i3_conn.get_tree()).find_focused())

        async def _callback_window(_, event: WindowEvent):
            match event.change:
                case "focus":
                    self._show_window(event.container)
                case "title":
                    if event.container.focused:
                        self._show_window(event.container)
                case "close" | "move":
                    # the focus may have moved to another window, or nowhere
                    await _show_focused()

        async def _callback_workspace(*_):
            # e.g. an empty workspace is focused
            await _show_focused()

        await _show_focused()

        i3_conn.on(Event.WINDOW, _callback_window)
        i3_conn.on(Event.WORKSPACE_FOCUS, _callback_workspace)

        try:
            await anyio.sleep_forever()
        finally:
            i3_conn.off(_callback_window)
            i3_conn.off(_callback_workspace)
