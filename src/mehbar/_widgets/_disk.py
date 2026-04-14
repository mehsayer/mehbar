from mehbar.widgets import Widget
import psutil

class WidgetDiskUsage(Widget):

    def __init__(self, interval: int, label_format: str, path: str):
        super().__init__(interval, label_format)
        self.path = path

    def update(self):
        dusage = psutil.disk_usage(self.path)

        if self._last_value != dusage.used:
            self._last_value = dusage.used

            self.format_label_idle(
                used_gib=round(dusage.used / (1024**3), 1),
                total_gib=round(dusage.total / (1024**3), 1),
                avail_gib=round(dusage.free / (1024**3), 1),
                percent=round(dusage.percent),
            )
