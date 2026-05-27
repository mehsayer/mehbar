from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


class WidgetCPUUsage(WidgetBase):
    TYPE = "cpu_usage"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self._last_percentage = -1

    async def run(self):

        from psutil import cpu_percent

        while await self.sleep_interval():
            percentage = min(round(cpu_percent()), 100)

            if self._last_percentage != percentage:
                self._last_percentage = percentage
                self.set_new_content_i(
                    ramp_level=percentage,
                    percent=percentage,
                )
