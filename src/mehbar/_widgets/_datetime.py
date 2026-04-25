from datetime import datetime

from mehbar.widgets import Widget


class WidgetDateTime(Widget):
    async def run(self):
        while await self.sleep_interval():
            self.format_label_idle(datetime=datetime.now())
