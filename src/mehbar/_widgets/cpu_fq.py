from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


class WidgetCPUFrequency(WidgetBase):
    TYPE = "cpu_fq"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self._last_percentage = -1

    async def run(self):
        from psutil import cpu_freq

        while await self.sleep_interval():
            fq = cpu_freq()

            # 'max' may be zero if cannot be determined,
            # 'current' may be more than 'max'
            percentage = round((fq.current / max(fq.current, fq.max, 1)) * 100)

            if self._last_percentage != percentage or percentage == 0:
                self._last_percentage = percentage

                self.set_new_content_i(
                    ramp_level=percentage,
                    percent=percentage,
                    fq_min=round(fq.min / 1000, 2),
                    fq_max=round(fq.max / 1000, 2),
                    fq=round(fq.current / 1000, 2),
                )
