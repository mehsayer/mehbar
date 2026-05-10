from functools import partial

import anyio

from mehbar._internals import BacklightACPI, BacklightDDCCI, BacklightInterface
from mehbar.exceptions import BarConfigError
from mehbar.widget import WidgetBase


class WidgetBacklight(WidgetBase):
    DRIVERS = {"acpi": BacklightACPI, "ddcci": BacklightDDCCI}
    UNIQUE = False
    TYPE = "backlight"

    def __init__(
        self,
        device: int | str,
        interval: int,
        label_format: str,
        ramp: list[str] | None = None,
        driver: str = "acpi",
        step: int = 10,
    ):
        super().__init__(interval, label_format, ramp)

        if driver not in self.DRIVERS:
            drivers = ", ".join(self.DRIVERS.keys())
            raise BarConfigError(f"{driver}: unknown backend, not one of: {drivers}.")

        self.driver = self.DRIVERS[driver](device)

        self.ramp = ramp
        self.ramps = []

        self.step = max(step, 1)

        self.sstream, self.rstream = anyio.create_memory_object_stream[int](8)

        self.onscroll_call(
            partial(self.elt_run_sync, self._change_level, self.step),
            partial(self.elt_run_sync, self._change_level, -self.step),
        )

    def _build_ramps(self, driver: BacklightInterface):

        if self.ramp is not None:
            max_level = driver.device.max_level

            for level in range(max_level + 1):
                if self.ramp and (nramp := len(self.ramp)) > 0:
                    ramp_idx = int(min(level, max_level - 1) / (max_level / nramp))
                    ramp_val = self.ramp[ramp_idx]

                    self.ramps.append(ramp_val)

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

        # while True:
        async with self.rstream:
            async for level in self.rstream:
                if level == 0:
                    display_level = await driver.get_level()
                else:
                    display_level = await driver.change_level(level)

                if display_level != self._last_value:
                    self._last_value = display_level

                    norm_level = int(max(0, min(display_level, max_level)))

                    ramp_ = None
                    if self.ramps:
                        ramp_ = self.ramps[norm_level]

                    self.format_label_idle(ramp=ramp_, level=norm_level)

    async def run(self):
        async with self.driver as bl_driver:
            self._build_ramps(bl_driver)

            async with anyio.create_task_group() as grp:
                grp.start_soon(self._poll, bl_driver)
                grp.start_soon(self._consume, bl_driver)
