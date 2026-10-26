from i3ipc import Event, WindowEvent

from mehbar.resource_manager import ResourceManager
from mehbar.widget import RewriteMixin, WidgetBase

from ._i3 import I3Connection, I3WidgetMixin


class WidgetI3Window(I3WidgetMixin, RewriteMixin, WidgetBase):
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

    async def i3_start(self, conn: I3Connection):
        async def _show_focused():
            self._show_window((await conn.get_tree()).find_focused())

        async def _callback_window(event: WindowEvent):
            match event.change:
                case "focus":
                    self._show_window(event.container)
                case "title":
                    if event.container.focused:
                        self._show_window(event.container)
                case "close" | "move":
                    # the focus may have moved to another window, or nowhere
                    await _show_focused()

        async def _callback_workspace(_event):
            # e.g. an empty workspace is focused
            await _show_focused()

        self.i3_on(conn, Event.WINDOW, _callback_window)
        self.i3_on(conn, Event.WORKSPACE_FOCUS, _callback_workspace)

        await _show_focused()
