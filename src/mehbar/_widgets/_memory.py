from mehbar.widgets import BarWidget
import psutil

class BarWidgetMemoryUsage(BarWidget):
    def update(self):
        vmem = psutil.virtual_memory()

        if self._last_value != vmem.used:
            self._last_value = vmem.used

            used_mib = vmem.used / (1024**2)
            total_mib = vmem.total / (1024**2)
            avail_mib = vmem.available / (1024**2)

            self.format_label_idle(
                used_mib=round(used_mib),
                used_gib=round(used_mib / 1024, 1),
                total_mib=round(total_mib),
                total_gib=round(total_mib / 1024, 1),
                avail_mib=round(avail_mib),
                avail_gib=round(avail_mib / 1024, 1),
                percent=round(vmem.percent),
            )
