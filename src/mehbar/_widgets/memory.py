from mehbar.widget import WidgetBase


class WidgetMemoryUsage(WidgetBase):
    TYPE = "memory_usage"

    async def run(self):
        from psutil import virtual_memory

        while await self.sleep_interval():
            vmem = virtual_memory()

            percent = min(round(vmem.percent), 100)

            used_mib = vmem.used / (1024**2)
            total_mib = vmem.total / (1024**2)
            avail_mib = vmem.available / (1024**2)

            self.set_new_content_i(
                percent,
                used_mib=round(used_mib),
                used_gib=round(used_mib / 1024, 1),
                total_mib=round(total_mib),
                total_gib=round(total_mib / 1024, 1),
                avail_mib=round(avail_mib),
                avail_gib=round(avail_mib / 1024, 1),
                percent=percent,
            )
