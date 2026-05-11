from pathlib import Path

import anyio
import psutil

from mehbar.exceptions import BarConfigError
from mehbar.widget import IconManager, WidgetBase, WidgetContent


class WidgetTemperature(WidgetBase):
    UNIQUE = False

    TYPE = "temperature"

    def __init__(
        self,
        label_format: str,
        interval: int = 15,
        ramp: list[str] | None = None,
        source: int | str | Path | None = 0,
        max_temp: int = 100,
        icon_manager: IconManager = None,
    ):
        self.max_temp = min(max_temp, 200)
        super().__init__(
            interval,
            label_format,
            ramp,
            icon_manager=icon_manager,
            max_ramp_level=self.max_temp,
        )

        expect_fields = ["ramp"]

        self.path_term = self.get_zone_path(source)

        self._coro_get_content = self._get_temp_file

        if self.path_term is None:
            self._coro_get_content = self._get_temp_sensors
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

    async def _get_temp_file(self) -> WidgetContent:
        temp = 0
        async with await anyio.open_file(self.path_term, "r") as fhandle:
            temp = int(await fhandle.readline()) // 1000

        norm_temp = min(temp, self.max_temp)
        return self.get_content(norm_temp, temp=norm_temp)

    async def _get_temp_sensors(self) -> WidgetContent:
        d_temps = {}
        max_curr_temp = 0

        for fld, temp in self.get_temperatures().items():
            if fld in self.fields:
                max_curr_temp = max(max_curr_temp, temp)
                d_temps[fld] = temp

        await anyio.lowlevel.checkpoint()

        norm_temp = min(max_curr_temp, self.max_temp)

        return self.get_content(norm_temp, **d_temps)

    async def run(self):
        while await self.sleep_interval():
            content = await self._coro_get_content()
            self.set_content_i(content)
