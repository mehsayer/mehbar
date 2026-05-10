import psutil

from mehbar.widget import WidgetBase


class WidgetCPUPercentage(WidgetBase):
    TYPE = "cpu_usage"

    async def run(self):
        while await self.sleep_interval():
            percentage = round(psutil.cpu_percent())

            if self._last_value != percentage:
                self._last_value = percentage
                self.format_label_idle(percent=percentage)
