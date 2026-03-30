from mehbar.widgets import BarWidget
from datetime import datetime

class BarWidgetDateTime(BarWidget):
    def update(self):
        self.format_label_idle(datetime=datetime.now())
