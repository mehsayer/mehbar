from pathlib import Path

import psutil

from mehbar.widget import WidgetBase


class WidgetDiskUsage(WidgetBase):
    TYPE = "disk"

    MAX_PERCENT = 100

    def __init__(
        self,
        interval: int,
        label_format: str,
        ramp: list[str] | None = None,
        path: str | Path = "/",
    ):
        super().__init__(interval, label_format, ramp, max_ramp_level=self.MAX_PERCENT)
        self.path = path
        self._last_used = -1

    async def run(self):
        while await self.sleep_interval():
            dusage = psutil.disk_usage(self.path)

            if self._last_used != dusage.used:
                self._last_used = dusage.used

                percent = min(round(dusage.percent), self.MAX_PERCENT)

                self.set_new_content_i(
                    ramp_level=percent,
                    used_gib=round(dusage.used / (1024**3), 1),
                    total_gib=round(dusage.total / (1024**3), 1),
                    avail_gib=round(dusage.free / (1024**3), 1),
                    percent=percent,
                )
