from mehbar.widget import WidgetBase


class WidgetCPUUsage(WidgetBase):
    TYPE = "cpu_usage"

    async def run(self):
        from psutil import cpu_percent

        # the first call returns a meaningless value, it starts the measurement
        cpu_percent()

        while await self.sleep_interval():
            percentage = min(round(cpu_percent()), 100)
            self.set_new_content_i(percentage, percent=percentage)
