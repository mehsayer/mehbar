import psutil

from mehbar.widgets import Widget


class WidgetCPUPercentage(Widget):
    async def run(self):
        while await self.sleep_interval():
            percentage = round(psutil.cpu_percent())

            if self._last_value != percentage:
                self._last_value = percentage
                self.format_label_idle(percent=percentage)
