from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


class WidgetDiskUsage(WidgetBase):
    TYPE = "disk_usage"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)
        self.path = self.cfg.get("path", "/")

    async def run(self):
        from psutil import disk_usage

        while await self.sleep_interval():
            dusage = disk_usage(self.path)

            percent = min(round(dusage.percent), 100)

            self.set_new_content_i(
                percent,
                used_gib=round(dusage.used / (1024**3), 1),
                total_gib=round(dusage.total / (1024**3), 1),
                avail_gib=round(dusage.free / (1024**3), 1),
                percent=percent,
            )
