from datetime import datetime

from gi.repository import Gtk  # type: ignore

from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


class WidgetDateTime(WidgetBase):
    """Ramp entries are spread over hours of the day. Updates happen at wall
    clock multiples of the interval, e.g. on the minute for 60 seconds. The
    tooltip is a calendar, unless the `tooltip` option is set."""

    TYPE = "datetime"
    POLLING = True
    RELAX_INTERVAL = False
    ALIGN_INTERVAL = True

    MAX_HOURS = 24

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.max_ramp_level = self.MAX_HOURS

    def build_tooltip(self) -> Gtk.Widget | None:
        if self.tooltip_content is not None or self._shown_tooltip:
            return super().build_tooltip()

        calendar = Gtk.Calendar()
        calendar.add_css_class("bar-tooltip")
        return calendar

    async def run(self):
        while await self.sleep_interval():
            now = datetime.now()
            self.set_new_content_i(now.hour, datetime=now)
