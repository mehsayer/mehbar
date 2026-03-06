from mehbar.widgets import BarWidget
import psutil

class BarWidgetCPUPercentage(BarWidget):
    def update(self):
        percentage = round(psutil.cpu_percent())

        if self._last_value != percentage:
            self._last_value = percentage
            self.format_label_idle(percent=percentage)
