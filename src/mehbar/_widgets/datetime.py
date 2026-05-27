from datetime import datetime

from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


class WidgetDateTime(WidgetBase):
    TYPE = "datetime"

    MAX_HOURS = 24

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.cfg["max_ramp_level"] = self.MAX_HOURS

    async def run(self):
        while await self.sleep_interval():
            now = datetime.now()
            self.set_new_content_i(ramp_level=now.hour, datetime=now)
