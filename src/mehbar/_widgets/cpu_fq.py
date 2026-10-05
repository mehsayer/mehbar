import errno

from mehbar.widget import WidgetBase


class WidgetCPUFrequency(WidgetBase):
    TYPE = "cpu_fq"

    async def run(self):
        from psutil import cpu_freq

        while await self.sleep_interval():
            if (fq := cpu_freq()) is None:
                raise OSError(errno.ENODEV, "CPU frequency is not available")

            # 'max' may be zero if cannot be determined,
            # 'current' may be more than 'max'
            percentage = round((fq.current / max(fq.current, fq.max, 1)) * 100)

            self.set_new_content_i(
                percentage,
                percent=percentage,
                fq_min=round(fq.min / 1000, 2),
                fq_max=round(fq.max / 1000, 2),
                fq=round(fq.current / 1000, 2),
            )
