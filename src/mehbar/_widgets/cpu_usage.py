import anyio

from mehbar.widget import WidgetBase


class WidgetCPUUsage(WidgetBase):
    TYPE = "cpu_usage"
    POLLING = True
    # the first measurement takes at least this long
    FIRST_SAMPLE = 0.5

    async def run(self):
        from psutil import cpu_percent

        # the first call returns a meaningless value, it starts the measurement
        cpu_percent()
        await anyio.sleep(self.FIRST_SAMPLE)

        while await self.sleep_interval():
            percentage = min(round(cpu_percent()), 100)
            self.set_new_content_i(percentage, percent=percentage)
