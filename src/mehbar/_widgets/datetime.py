from datetime import datetime

from mehbar.widget import WidgetBase


class WidgetDateTime(WidgetBase):
    TYPE = "datetime"

    async def run(self):
        while await self.sleep_interval():
            self.set_new_content_i(datetime=datetime.now())
