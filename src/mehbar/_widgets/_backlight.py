from mehbar.widgets import Widget
from mehbar.exceptions import BarConfigError

from mehbar._internals import BacklightDDCCI, BacklightInterface, BacklightACPI
from functools import partial
import asyncio

class WidgetBacklight(Widget):

    DRIVERS = {
        "acpi": BacklightACPI,
        "ddcci": BacklightDDCCI
    }

    ACT_DISPLAY = 0
    ACT_SET = 1

    def __init__(self,
        driver: str,
        device: int | str,
        step: int,
        interval: int,
        label_format: str,
        ramp: list[str] | None = None):
        super().__init__(interval, label_format, ramp)

        if driver not in self.DRIVERS:
            drivers = ', '.join(self.DRIVERS.keys())
            raise BarConfigError(f"{driver}: unknown backend, not one of: {drivers}.")

        self.driver = self.DRIVERS[driver](device)

        self.ramp = ramp

        self.step = max(step, 1)

        self.queue = asyncio.Queue(maxsize=8)

        self.aio_loop = None

        self.ramps = []

        self.onscroll_call(partial(self._change_level_threadsafe, self.step),
                           partial(self._change_level_threadsafe, -self.step))


    async def _build_ramps(self, driver: BacklightInterface):

        max_level = driver.device.max_level

        for level in range(max_level + 1):
            ramp_val = str()

            if self.ramp is not None and (nramp := len(self.ramp)) > 0:
                ramp_idx = int(min(level, max_level - 1) / (max_level / nramp))
                ramp_val = self.ramp[ramp_idx]

            self.ramps.append(ramp_val)


    def _change_level_threadsafe(self, value: int):
        if self.aio_loop is not None:
            self.aio_loop.call_soon_threadsafe(self._change_level, value)


    def _change_level(self, value: int):
        if not self.queue.full():
            self.queue.put_nowait((self.ACT_SET, value))


    async def _poll(self, driver: BacklightInterface):
        while True:
            if not self.queue.full():
                self.queue.put_nowait((self.ACT_DISPLAY, await driver.get_level()))

            await asyncio.sleep(self.interval)


    async def _consume(self, driver: BacklightInterface):

        display_level = 0

        max_level = driver.device.max_level

        while True:
            source, level = await self.queue.get()

            if source == self.ACT_DISPLAY:
                display_level = level
            else:
                display_level = await driver.change_level(level)

            if display_level != self._last_value:
                self._last_value = display_level

                norm_level=int(max(0, min(display_level, max_level)))

                self.format_label_idle(ramp=self.ramps[norm_level], level=norm_level)


    async def run(self):

        self.aio_loop = asyncio.get_running_loop()

        async with self.driver as bl_driver:

            await self._build_ramps(bl_driver)

            async with asyncio.TaskGroup() as grp:
                tlisten = grp.create_task(self._poll(bl_driver))
                tconsume = grp.create_task(self._consume(bl_driver))
