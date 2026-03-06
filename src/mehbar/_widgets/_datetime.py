from mehbar.widgets import BarWidget
from datetime import datetime

class BarWidgetDateTime(BarWidget):
    def __init__(self, interval: int, label_format: str, datetime_format: str):
        super().__init__(interval, label_format)
        self.datetime_format = datetime_format

    def update(self):
        datetime_str = datetime.now().strftime(self.datetime_format)

        if self._last_value != datetime_str:
            self._last_value = datetime_str
            self.format_label_idle(datetime=datetime_str)
