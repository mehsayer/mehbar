from datetime import datetime

from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


class WidgetDateTime(WidgetBase):
    """Ramp entries are spread over hours of the day. Updates happen at wall
    clock multiples of the interval, e.g. on the minute for 60 seconds."""

    TYPE = "datetime"
    RELAX_INTERVAL = False
    ALIGN_INTERVAL = True

    MAX_HOURS = 24

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.max_ramp_level = self.MAX_HOURS

    async def run(self):
        while await self.sleep_interval():
            now = datetime.now()
            self.set_new_content_i(now.hour, datetime=now)
