import logging
from functools import partial

import anyio

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase

from ._backlight import BacklightACPI, BacklightDDCCI, BacklightInterface


class WidgetBacklight(WidgetBase):
    DRIVERS = {"acpi": BacklightACPI, "ddcci": BacklightDDCCI}
    UNIQUE = False
    TYPE = "backlight"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        driver = self.cfg.get("driver")
        device = self.cfg.get("device")

        logging.debug("DASDA")

        if driver is None:
            raise BarConfigError("no driver specified in configuration")

        if driver not in self.DRIVERS:
            drivers = ", ".join([f"'{s}'" for s in self.DRIVERS.keys()])
            raise BarConfigError(f"{driver}: unknown driver, not one of: {drivers}.")

        self.driver = self.DRIVERS[driver](device)

        self.step = max(self.cfg.get("step", 1), 1)

        self.sstream, self.rstream = anyio.create_memory_object_stream[int](8)

        self.onscroll_call(
            partial(self.elt_run_sync, self._change_level, self.step),
            partial(self.elt_run_sync, self._change_level, -self.step),
        )

    def _change_level(self, level: int):
        try:
            self.sstream.send_nowait(level)
        except anyio.WouldBlock:
            pass

    async def _poll(self, driver: BacklightInterface):

        while await self.sleep_interval():
            try:
                self.sstream.send_nowait(0)
            except anyio.WouldBlock:
                pass

    async def _consume(self, driver: BacklightInterface):

        display_level = 0

        max_level = driver.device.max_level

        async with self.rstream:
            async for level in self.rstream:
                if level == 0:
                    display_level = await driver.get_level()
                else:
                    display_level = await driver.change_level(level)

                if display_level != self._last_value:
                    self._last_value = display_level

                    percent = int(max(0, min(display_level, max_level)))

                    self.set_new_content_i(ramp_level=percent, percent=percent)

    async def run(self):
        async with self.driver as bl_driver:
            async with anyio.create_task_group() as grp:
                grp.start_soon(self._poll, bl_driver)
                grp.start_soon(self._consume, bl_driver)
