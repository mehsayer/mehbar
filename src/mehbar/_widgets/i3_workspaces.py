import logging
import re
from dataclasses import dataclass

import anyio
from gi.repository import Gtk  # type: ignore

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import BarWidget, IdleUpdater, RewriteMixin, WidgetBase


@dataclass(frozen=True)
class WorkspaceState:
    name: str
    num: int
    label: str
    exists: bool = True
    focused: bool = False
    visible: bool = False
    urgent: bool = False
    previous: bool = False

    @property
    def ramp_level(self) -> int:
        """Ramp entries are for: empty, normal, focused, urgent."""
        if self.urgent:
            return 3
        if self.focused:
            return 2
        return 1 if self.exists else 0

    @property
    def css_classes(self) -> dict[str, bool]:
        return {
            "focused": self.focused,
            "visible": self.visible,
            "urgent": self.urgent,
            "previous": self.previous,
            "empty": not self.exists,
        }


class I3WorkspaceButton(WidgetBase):
    def __init__(self, ws_name: str, parent: "WidgetI3Workspaces"):
        super().__init__(
            ws_name,
            parent.res_mgr,
            {"label": parent.button_label, "ramp": parent.button_ramp},
        )

        self.ws_name = ws_name
        self.add_css_class("i3-workspace")
        self.onclick_call(1, self.run_soon, parent.switch_to, ws_name)

    def ramp_index(self, ramp_level: int) -> int | None:
        if ramp_level < 0 or not self.ramp:
            return None
        return min(ramp_level, len(self.ramp) - 1)

    def update(self, state: WorkspaceState):
        """Must be called from GTK main thread."""
        self.apply_content(
            self.get_content(state.ramp_level, name=state.label, num=state.num)
        )

        for css_class, enabled in state.css_classes.items():
            if enabled:
                self.add_css_class(css_class)
            else:
                self.remove_css_class(css_class)


class WidgetI3Workspaces(RewriteMixin, BarWidget):
    """Shows workspace buttons. Button labels are formatted using `label`
    with `name` (rewritten) and `num` fields, ramp entries are for workspace
    states: empty, normal, focused, urgent. Workspaces listed in
    `always_show` are shown even if they do not exist."""

    MAX_WORKSPACE_CNT = 20
    MAX_SCROLL_SPEED = 100
    DEFAULT_MAX_WORKSPACES = 10
    DEFAULT_SCROLL_SPEED = 10

    TYPE = "i3_workspaces"

    RE_NUM = re.compile(r"^(\d+)")

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        always_show = self.cfg.get("always_show") or []

        if not isinstance(always_show, list):
            raise BarConfigError("'always_show' must be a list of workspace names")

        self.always_show = [str(ws_name) for ws_name in always_show]

        max_workspaces = self.cfg.get("max_workspaces", self.DEFAULT_MAX_WORKSPACES)
        self.max_workspaces = max(1, min(max_workspaces, self.MAX_WORKSPACE_CNT))

        self.button_label = self.cfg.get("label", "{name}")
        self.button_ramp = self.cfg.get("ramp") or []

        scroll_width = self.cfg.get("scroll_width", 0)
        scroll_speed = self.cfg.get("scroll_speed", self.DEFAULT_SCROLL_SPEED)

        self.box = Gtk.Box.new(Gtk.Orientation.HORIZONTAL, 0)
        self.scroller = Gtk.ScrolledWindow.new()
        self.scroller.set_propagate_natural_height(True)
        self.scroller.set_has_frame(False)
        self.scroller.set_kinetic_scrolling(False)
        self.viewport = None

        if scroll_width > 0:
            scroll_speed = max(1, min(scroll_speed, self.MAX_SCROLL_SPEED))
            self.scroller.set_min_content_width(scroll_width)
            self.scroller.set_size_request(scroll_width, -1)
            self.scroller.set_policy(Gtk.PolicyType.EXTERNAL, Gtk.PolicyType.NEVER)
            self.viewport = Gtk.Viewport.new()
            self.viewport.set_child(self.box)
            self.viewport.set_scroll_to_focus(False)

            h_adj = self.scroller.get_hadjustment()

            def _scroll(_ctrl, _dx, dy):
                h_adj.set_value(h_adj.get_value() + (dy * scroll_speed))
                return True

            scroll_ctrl = Gtk.EventControllerScroll.new(
                Gtk.EventControllerScrollFlags.VERTICAL
            )
            scroll_ctrl.connect("scroll", _scroll)
            self.viewport.add_controller(scroll_ctrl)
            self.scroller.set_child(self.viewport)
        else:
            self.scroller.set_propagate_natural_width(True)
            self.scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.NEVER)
            self.scroller.set_child(self.box)

        self.append(self.scroller)

        # GTK main thread only
        self._buttons: dict[str, I3WorkspaceButton] = {}
        self._updater = IdleUpdater(self._apply_states)

        # event loop thread only
        self._i3_conn = None
        self._prev_focus: str | None = None
        self._refresh_seq = 0
        self._too_many_logged = False

    def _num_for_name(self, ws_name: str) -> int:
        match = self.RE_NUM.match(ws_name)
        return int(match.group(1)) if match else -1

    # Event loop thread

    async def switch_to(self, ws_name: str):
        i3_conn = await self.res_mgr.get_i3_connection_async()
        escaped = ws_name.replace("\\", "\\\\").replace('"', '\\"')
        await i3_conn.command(f'workspace "{escaped}"')

    async def _refresh(self):
        self._refresh_seq += 1
        seq = self._refresh_seq

        workspaces = await self._i3_conn.get_workspaces()

        # a later refresh has finished first
        if seq != self._refresh_seq:
            return

        states = {}

        for ws in workspaces:
            states[ws.name] = WorkspaceState(
                ws.name,
                ws.num,
                self.rewrite(ws.name),
                focused=ws.focused,
                visible=ws.visible and not ws.focused,
                urgent=ws.urgent,
                previous=ws.name == self._prev_focus and not ws.focused,
            )

        for ws_name in self.always_show:
            if ws_name not in states:
                states[ws_name] = WorkspaceState(
                    ws_name,
                    self._num_for_name(ws_name),
                    self.rewrite(ws_name),
                    exists=False,
                    previous=ws_name == self._prev_focus,
                )

        # numbered workspaces first, like i3 does
        ordered = sorted(
            states.values(), key=lambda s: (s.num < 0, s.num, s.name)
        )

        if len(ordered) > self.max_workspaces:
            if not self._too_many_logged:
                self._too_many_logged = True
                logging.warning(
                    "widget '%s': showing only %d workspaces",
                    self.widget_name,
                    self.max_workspaces,
                )
            ordered = ordered[: self.max_workspaces]

        self._updater.submit(tuple(ordered))

    async def run(self):
        from i3ipc import Event, WorkspaceEvent

        self._i3_conn = await self.res_mgr.get_i3_connection_async()

        async def _callback_workspace(_, event: WorkspaceEvent):
            if event.change == "focus" and event.old is not None:
                self._prev_focus = event.old.name
            await self._refresh()

        await self._refresh()

        self._i3_conn.on(Event.WORKSPACE, _callback_workspace)

        try:
            await anyio.sleep_forever()
        finally:
            self._i3_conn.off(_callback_workspace)

    # GTK main thread

    def _apply_states(self, states: tuple[WorkspaceState, ...]):
        wanted = {state.name for state in states}

        for ws_name in list(self._buttons):
            if ws_name not in wanted:
                self.box.remove(self._buttons.pop(ws_name))

        prev_button = None
        focused_button = None

        for state in states:
            if (button := self._buttons.get(state.name)) is None:
                button = I3WorkspaceButton(state.name, self)
                self._buttons[state.name] = button
                self.box.insert_child_after(button, prev_button)
            elif button.get_prev_sibling() is not prev_button:
                self.box.reorder_child_after(button, prev_button)

            button.update(state)

            if state.focused:
                focused_button = button

            prev_button = button

        if self.viewport is not None and focused_button is not None:
            self.viewport.scroll_to(focused_button, None)
