import psutil

from mehbar.widget import IconManager, WidgetBase


class WidgetMemoryUsage(WidgetBase):
    TYPE = "memory"

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
            vmem = psutil.virtual_memory()

            if self._last_percentage != vmem.used:
                self._last_percentage = vmem.used

                percent = min(round(vmem.percent), self.MAX_PERCENT)

                used_mib = vmem.used / (1024**2)
                total_mib = vmem.total / (1024**2)
                avail_mib = vmem.available / (1024**2)

                self.set_new_content_i(
                    ramp_level=percent,
                    used_mib=round(used_mib),
                    used_gib=round(used_mib / 1024, 1),
                    total_mib=round(total_mib),
                    total_gib=round(total_mib / 1024, 1),
                    avail_mib=round(avail_mib),
                    avail_gib=round(avail_mib / 1024, 1),
                    percent=percent,
                )
