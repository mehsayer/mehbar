from mehbar.widgets import Widget
import psutil

class WidgetCPUPercentage(Widget):
    def update(self):
        percentage = round(psutil.cpu_percent())

        if self._last_value != percentage:
            self._last_value = percentage
            self.format_label_idle(percent=percentage)
