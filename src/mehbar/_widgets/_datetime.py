from mehbar.widgets import Widget
from datetime import datetime

class WidgetDateTime(Widget):
    def update(self):
        self.format_label_idle(datetime=datetime.now())
