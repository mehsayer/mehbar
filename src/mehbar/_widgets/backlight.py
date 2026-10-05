from functools import partial

import anyio
from anyio.abc import ObjectReceiveStream, ObjectSendStream

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase

from ._backlight import BacklightACPI, BacklightDDCCI, BacklightInterface


class WidgetBacklight(WidgetBase):
    """Shows and, on scroll, changes the display brightness in percent."""

    DRIVERS = {"acpi": BacklightACPI, "ddcci": BacklightDDCCI}
    UNIQUE = False
    TYPE = "backlight"
    QUEUE_SIZE = 8

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

        self._sstream: ObjectSendStream[int] | None = None

        self.onscroll_call(
            partial(self.call_soon, self._request, self.step),
            partial(self.call_soon, self._request, -self.step),
        )

    def _request(self, delta: int):
        """Requests a brightness change by `delta`, 0 to refresh."""
        if self._sstream is not None:
            try:
                self._sstream.send_nowait(delta)
            except (anyio.WouldBlock, anyio.ClosedResourceError):
                pass

    async def _poll(self):
        while await self.sleep_interval():
            self._request(0)

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
        self._sstream, rstream = anyio.create_memory_object_stream[int](
            self.QUEUE_SIZE
        )

        try:
            async with self.driver as driver, anyio.create_task_group() as grp:
                grp.start_soon(self._poll)
                grp.start_soon(self._consume, driver, rstream)
        finally:
            self._sstream.close()
            self._sstream = None
