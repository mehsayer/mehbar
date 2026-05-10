from datetime import datetime

from mehbar.widget import WidgetBase


class WidgetDateTime(WidgetBase):
    TYPE = "datetime"

    async def run(self):
        while await self.sleep_interval():
            self.format_label_idle(datetime=datetime.now())
