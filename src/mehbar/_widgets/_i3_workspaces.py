from mehbar.widgets import BarWidget, BarWindgetInterface, RewriteMixin, I3ListenerMixin
from gi.repository import Gtk, GLib
from itertools import compress
from i3ipc.aio import Connection
from i3ipc import Event
import asyncio

class I3WorkspaceButton(I3ListenerMixin, BarWidget):
    def __init__(self, name: str, label: str, i3_conn: Connection):
        super().__init__(0, None, i3_conn=i3_conn)
        self.set_name(name)
        self.set_label(label)
        self.add_css_class("workspace")
        self.onclick_call(1, self.switch_ws)
        self.aio_loop = asyncio.get_running_loop()

    async def _switch_ws_async(self, name: str):
        i3_conn = await self.get_i3_conn()
        return await i3_conn.command("workspace " + name)

    def switch_ws(self):
        name = self.get_name()
        self.aio_loop.call_soon_threadsafe(asyncio.ensure_future,
                                           self._switch_ws_async(name))


class BarWidgetI3Workspaces(I3ListenerMixin,
                            RewriteMixin,
                            Gtk.ScrolledWindow,
                            BarWindgetInterface):

    MAX_WORKSPACES = 20
    MAX_SCROLL_SPEED = 100

    def __init__(self,
                 rewrite: dict[str, str],
                 i3_conn: Connection,
                 scroll_width: int,
                 scroll_speed: int,
                 max_workspaces: int,
                 always_show: list[str] | None = None,
                 **kwargs):
        super().__init__(i3_conn=i3_conn, rewrite=rewrite)

        self.wsid_map: dict[str, int] = {}
        self.ws_button_map: dict[str, I3WorkspaceButton] = {}

        self.always_show = []
        if always_show is not None:
            self.always_show.extend([str(name) for name in always_show])

        self.max_workspaces = max(1, min(max_workspaces, self.MAX_WORKSPACES))

        self.set_propagate_natural_height(True)
        self.set_has_frame(False)
        self.set_kinetic_scrolling(False)

        self.box = Gtk.Box.new(Gtk.Orientation.HORIZONTAL, 0)

        self.cache: dict[str, Any] = {}
        self.viewport = None
        self.cur_focus = None
        self.prev_focus = None
        self.i3_conn = None

        if scroll_width > 0:
            scroll_speed = max(1, min(scroll_speed, self.MAX_SCROLL_SPEED))
            self.set_min_content_width(scroll_width)
            self.set_size_request(scroll_width, -1)
            self.set_policy(Gtk.PolicyType.EXTERNAL, Gtk.PolicyType.NEVER)
            self.viewport = Gtk.Viewport.new()
            self.viewport.set_child(self.box)

            self.h_adj = self.get_hadjustment()

            def _scroll(ctrl, _, direction):
                self.h_adj.set_value(
                    self.h_adj.get_value() + (direction * scroll_speed)
                )

            scroll_ctrl = Gtk.EventControllerScroll.new(
                Gtk.EventControllerScrollFlags.VERTICAL
            )
            scroll_ctrl.connect("scroll", _scroll)
            self.viewport.add_controller(scroll_ctrl)
            self.viewport.set_scroll_to_focus(False)
            self.set_child(self.viewport)
        else:
            self.set_propagate_natural_width(True)
            self.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.NEVER)
            self.set_child(self.box)

    def scroll_into_view(self, widget: Gtk.Widget):
        if self.viewport is not None:
            self.viewport.scroll_to(widget)


    def rewrite(self, text: str) -> str:
        if text not in self.cache:
            self.cache[text] = super().rewrite(text)
        return self.cache[text]


    def set_empty_idle(self, child: Gtk.Widget) -> None:
        child.set_visible(False)
        child.remove_css_class("focused")
        child.remove_css_class("previous")
        child.remove_css_class("urgent")
        return GLib.SOURCE_REMOVE

    def set_focused_idle(self, child: Gtk.Widget) -> None:
        child.add_css_class("focused")
        child.set_visible(True)
        self.scroll_into_view(child)
        return GLib.SOURCE_REMOVE

    def set_urgent_idle(self, child: Gtk.Widget) -> None:
        if child.has_css_class("urgent"):
            child.remove_css_class("urgent")
        else:
            child.add_css_class("urgent")
            child.set_visible(True)
            self.scroll_into_view(child)
        return GLib.SOURCE_REMOVE

    def add_css_class_idle(self, child: Gtk.Widget, css_class: str) -> None:
        child.add_css_class(css_class)
        return GLib.SOURCE_REMOVE

    def remove_css_class_idle(self, child: Gtk.Widget, css_class: str) -> None:
        child.remove_css_class(css_class)
        return GLib.SOURCE_REMOVE

    def set_name_idle(self, child: Gtk.Widget, name: str) -> None:
        child.set_name(name)
        return GLib.SOURCE_REMOVE

    def set_label_idle(self, child: Gtk.Widget, label: str) -> None:
        child.set_label(label)
        return GLib.SOURCE_REMOVE


    def dispatch_ws(self,
                    name: str,
                    old_focus_name: str,
                    action: str,
                    wsid: int) -> None:

        child = None
        old_name = None
        label = self.rewrite(name)

        if self.wsid_map.get(name, -1) < 0 and wsid >= 0:
            if action == "rename":
                for _name, _wsid in self.wsid_map.items():
                    if _wsid == wsid:
                        old_name = _name
                        break

                if old_name in self.wsid_map:
                    self.wsid_map[name] = self.wsid_map.pop(old_name)
                else:
                    self.wsid_map[name] = wsid

        if old_name in self.ws_button_map:
            child = self.ws_button_map[old_name]
            GLib.idle_add(self.set_name_idle, child, name)
            GLib.idle_add(self.set_label_idle, child, label)
        elif name in self.ws_button_map:
            child = self.ws_button_map[name]

            match action:
                case "empty":
                    if name not in self.always_show:
                        GLib.idle_add(self.set_empty_idle, child)
                    else:
                        GLib.idle_add(self.remove_css_class_idle, child, "focused")
                    GLib.idle_add(self.remove_css_class_idle, child, "urgent")
                case "focus":
                    if self.cur_focus in self.ws_button_map:
                        GLib.idle_add(self.remove_css_class_idle, self.ws_button_map[self.cur_focus], "focused")
                        GLib.idle_add(self.add_css_class_idle, self.ws_button_map[old_focus_name], "previous")

                    if self.prev_focus in self.ws_button_map:
                        GLib.idle_add(self.remove_css_class_idle, self.ws_button_map[self.prev_focus], "previous")

                    GLib.idle_add(self.set_focused_idle, child)

                    self.prev_focus, self.cur_focus = self.cur_focus, name
                case "urgent":
                    GLib.idle_add(self.set_urgent_idle, child)
                case _:
                    pass
        elif len(self.ws_button_map) < self.max_workspaces:
            child = I3WorkspaceButton(name, label, self.i3_conn)
            self.box.append(child)
            self.ws_button_map[name] = child
            if wsid >= 0:
                self.wsid_map[name] = wsid
        else:
            logging.error("refusing to track more than %d workspaces", self.max_workspaces)

        return GLib.SOURCE_REMOVE

    async def run(self):

        self.i3_conn = await self.get_i3_conn()

        async def _callback_workspaces(_, event: WorkspaceEvent):
            if event.change not in ["move", "restore", "reload"]:
                prev_name = None
                if event.old is not None:
                    prev_name = event.old.name

                self.dispatch_ws(event.current.name,
                    prev_name,
                    event.change,
                    event.current.id,
                )

        existing_ws = {}
        for ws in await self.i3_conn.get_workspaces():
            existing_ws[ws.name] = (ws.ipc_data["id"], ws.focused, ws.urgent)

        for name in self.always_show:
            wsid, *_ = existing_ws.get(name, (-1, False, False))

            if name is not None:
                self.dispatch_ws(name, None, "init", wsid)

        actions = ["init", "focus", "urgent"]
        for name, (wsid, focus, urgent) in existing_ws.items():
            for action in compress(actions, [True, focus, urgent]):
                self.dispatch_ws(name, None, action, wsid)

        self.i3_conn.on(Event.WORKSPACE, _callback_workspaces)
