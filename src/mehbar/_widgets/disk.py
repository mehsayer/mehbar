from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


class WidgetDiskUsage(WidgetBase):
    TYPE = "disk_usage"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)
        self.path = self.cfg.get("path", "/")
        self._last_used = -1

    async def run(self):
        from psutil import disk_usage

        while await self.sleep_interval():
            dusage = disk_usage(self.path)

            if self._last_used != dusage.used:
                self._last_used = dusage.used

                percent = min(round(dusage.percent), 100)

                self.set_new_content_i(
                    ramp_level=percent,
                    used_gib=round(dusage.used / (1024**3), 1),
                    total_gib=round(dusage.total / (1024**3), 1),
                    avail_gib=round(dusage.free / (1024**3), 1),
                    percent=percent,
                )
