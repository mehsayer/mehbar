import os
from pathlib import Path

import anyio
import psutil

from mehbar.exceptions import BarConfigError
from mehbar.widgets import Widget


class WidgetTemperature(Widget):
    def __init__(
        self,
        source: int | str | Path | None,
        max_temp: int,
        interval: int,
        label_format: str,
        ramp: list[str] | None = None,
    ):
        super().__init__(interval, label_format, ramp)

        expect_fields = ["ramp"]

        self.path_term = self.get_zone_path(source)

        self.coro_temp_getter = self._get_temp_file

        if self.path_term is None:
            self.coro_temp_getter = self._get_temp_sensors
            expect_fields.extend(self.get_temperatures().keys())
        elif not self.path_term.is_file():
            raise BarConfigError(f"source file does not exist: {self.path_term}")
        else:
            expect_fields.append("temp")

        self.fields = []

        for fld in set(self.formatter.get_fields(label_format)):
            if fld not in expect_fields:
                raise BarConfigError(f"unknown label field: {fld}")
            else:
                self.fields.append(fld)

        self.max_temp = max(min(max_temp, 200), 0)
        self.ramps = []

        for temp in range(self.max_temp + 1):
            ramp_val = str()

            if ramp is not None and (nramp := len(ramp)) > 0:
                ramp_idx = int(min(temp, self.max_temp - 1) / (self.max_temp / nramp))
                ramp_val = ramp[ramp_idx]

            self.ramps.append(ramp_val)

    def get_zone_path(self, source: int | str | Path | None) -> Path:
        n_zone = 0

        zone_path = None

        if source is not None:
            try:
                n_zone = int(source)
            except ValueError:
                if isinstance(source, str):
                    source_path = Path(source)
                    if source_path.is_absolute():
                        zone_path = source_path
                elif isinstance(source, Path):
                    zone_path = source
            else:
                zone_path = Path(f"/sys/class/thermal/thermal_zone{n_zone}/temp")
        return zone_path

    def get_temperatures(self) -> dict[str, int]:
        d_temps = {}
        for name, l_swhtemp in psutil.sensors_temperatures().items():
            for swhtemp in l_swhtemp:
                selector = name
                if swhtemp.label:
                    selector += "/" + swhtemp.label

                d_temps[selector] = round(swhtemp.current)
        return d_temps

    async def _get_temp_file(self) -> str:
        temp = 0

        async with await anyio.open_file(self.path_term, "r") as fhandle:
            temp = int(await fhandle.readline()) // 1000

        norm_temp = min(temp, self.max_temp)
        return self.vsformat(ramp=self.ramps[norm_temp], temp=norm_temp)

    async def _get_temp_sensors(self) -> str:
        d_temps = {}
        max_curr_temp = 0

        for fld, temp in self.get_temperatures().items():
            if fld in self.fields:
                max_curr_temp = max(max_curr_temp, temp)
                d_temps[fld] = temp

        await anyio.lowlevel.checkpoint()

        norm_temp = min(max_curr_temp, self.max_temp)

        return self.vsformat(ramp=self.ramps[norm_temp], **d_temps)

    async def run(self):
        while await self.sleep_interval():
            label_str = await self.coro_temp_getter()
            self.set_label_idle(label_str)
