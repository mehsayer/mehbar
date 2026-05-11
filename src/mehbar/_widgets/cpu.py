import psutil

from mehbar.widget import IconManager, WidgetBase


class WidgetCPUPercentage(WidgetBase):
    TYPE = "cpu_usage"

    MAX_PERCENT = 100

    def __init__(
        self,
        interval: int = 0,
        label_format: str | None = None,
        ramp: list[str] | None = None,
        icon_manager: IconManager | None = None,
    ):
        super().__init__(
            interval, label_format, ramp, icon_manager, max_ramp_level=self.MAX_PERCENT
        )

        self._last_percentage = -1

    async def run(self):
        while await self.sleep_interval():
            percentage = min(round(psutil.cpu_percent()), self.MAX_PERCENT)

            if self._last_percentage != percentage:
                self._last_percentage = percentage
                self.set_new_content_i(ramp_level=percentage, percent=percentage)
