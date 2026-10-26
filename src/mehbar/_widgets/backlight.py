from functools import partial

import anyio
from anyio.abc import ObjectReceiveStream

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import RequestQueue, WidgetBase

from ._backlight import BacklightACPI, BacklightDDCCI, BacklightInterface


class WidgetBacklight(WidgetBase):
    """Shows and, on scroll, changes the display brightness in percent."""

    DRIVERS = {"acpi": BacklightACPI, "ddcci": BacklightDDCCI}
    UNIQUE = False
    TYPE = "backlight"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        driver = self.cfg.get("driver")
        device = self.cfg.get("device")

        if driver is None:
            raise BarConfigError("no driver specified in configuration")

        if driver not in self.DRIVERS:
            drivers = ", ".join([f"'{s}'" for s in self.DRIVERS.keys()])
            raise BarConfigError(f"{driver}: unknown driver, not one of: {drivers}.")

        self.driver = self.DRIVERS[driver](device)

        self.step = max(self.cfg.get("step", 1), 1)

        # brightness changes by a delta, 0 refreshes
        self.requests = RequestQueue(res_mgr.loop)

        self.onscroll_call(
            partial(self.requests.put, self.step),
            partial(self.requests.put, -self.step),
        )

    async def _poll(self):
        while await self.sleep_interval():
            self.requests.put(0)

    async def _consume(
        self, driver: BacklightInterface, rstream: ObjectReceiveStream[int]
    ):
        async with rstream:
            async for delta in rstream:
                if delta == 0:
                    level = await driver.get_level()
                else:
                    level = await driver.change_level(delta)

                percent = round(max(0, min(level, 100)))
                self.set_new_content_i(percent, percent=percent)

    async def run(self):
        with self.requests.receive() as rstream:
            async with self.driver as driver, anyio.create_task_group() as grp:
                grp.start_soon(self._poll)
                grp.start_soon(self._consume, driver, rstream)
