import logging

from gi.repository import Gdk, Gtk  # type: ignore

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


class WidgetIdleInhibitor(WidgetBase):
    """Keeps the screen from blanking and locking while it is on, a click
    turns it on or off. Ramp entries are for the state: off, on. The `active`
    option turns it on at start. The compositor must support the Wayland
    idle inhibit protocol, sway does."""

    TYPE = "idle_inhibitor"
    STATIC = True
    RAMP_BY_STATE = True
    DEFAULT_TOOLTIP = "Idle inhibitor: {status}"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        if not isinstance(active := self.cfg.get("active", False), bool):
            raise BarConfigError("'active' must be true or false")

        # GtkApplication inhibit cookie, 0 while off
        self._cookie = 0

        self.onclick_call(Gdk.BUTTON_PRIMARY, self.toggle)
        self._show()

        if active:
            # inhibiting needs the surface of the window, once
            def _on_map(*_):
                self.disconnect(handler_id)
                if not self.active:
                    self.toggle()

            handler_id = self.connect("map", self.guard(_on_map))

    @property
    def active(self) -> bool:
        return self._cookie != 0

    def toggle(self):
        window = self.get_root()
        app = window.get_application() if isinstance(window, Gtk.Window) else None

        if app is None:
            return

        if self.active:
            app.uninhibit(self._cookie)
            self._cookie = 0
        else:
            self._cookie = app.inhibit(
                window, Gtk.ApplicationInhibitFlags.IDLE, "Idle inhibitor of mehbar"
            )

            if not self.active:
                logging.warning(
                    "widget '%s': idle cannot be inhibited", self.widget_name
                )

        self._show()

    def _show(self):
        self.apply_content(
            self.get_content(int(self.active), status="on" if self.active else "off")
        )

    def teardown(self):
        if self.active and (window := self.get_root()) is not None:
            if (app := window.get_application()) is not None:
                app.uninhibit(self._cookie)
        self._cookie = 0
        super().teardown()
