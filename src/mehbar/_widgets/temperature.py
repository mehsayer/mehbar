from pathlib import Path

import anyio

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase, WidgetContent


class WidgetTemperature(WidgetBase):
    UNIQUE = False

    TYPE = "temperature"

    MAX_TEMP_LO = 100
    MAX_TEMP_HI = 150

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        exp_flds = ["ramp"]

        self.max_temp = min(
            self.cfg.get("max_temp", self.MAX_TEMP_LO), self.MAX_TEMP_HI
        )

        self.cfg.setdefault("max_ramp_level", self.max_temp)

        source = self.cfg.get("path")

        self.path_term = None

        # TOML does not have a proper 'null' value
        if source is not None and source.lower() not in ["none", "null"]:
            self.path_term = Path(source)

        if self.path_term is None:
            self._coro_get_content = self._get_temp_sensors
            exp_flds.extend(self.get_temperatures().keys())
        elif not self.path_term.is_file():
            raise OSError(f"source file does not exist: {self.path_term}")
        else:
            self._coro_get_content = self._get_temp_file
            exp_flds.append("temp")

        self.fields = []

        unk_flds = set()

        for fld in set(self.formatter.get_fields(self.content.label)):
            if fld not in exp_flds:
                unk_flds.add(fld)
            else:
                self.fields.append(fld)

        if unk_flds:
            unk_flds_s = ", ".join([repr(fld) for fld in unk_flds])
            exp_flds_s = ", ".join(repr(fld) for fld in exp_flds if fld != "ramp")
            raise BarConfigError(
                f"unknown fields: {unk_flds_s}; expected one or more of: {exp_flds_s}"
            )

    def get_temperatures(self) -> dict[str, int]:

        from psutil import sensors_temperatures

        d_temps = {}
        for name, l_swhtemp in sensors_temperatures().items():
            for swhtemp in l_swhtemp:
                selector = name
                if swhtemp.label:
                    selector += "/" + swhtemp.label

                d_temps[selector] = round(swhtemp.current)
        return d_temps

    async def _get_temp_file(self) -> WidgetContent:
        temp = 0
        if self.path_term is not None:
            async with await anyio.open_file(self.path_term, "r") as fhandle:
                temp = int(await fhandle.readline()) // 1000

        norm_temp = min(temp, self.max_temp)
        return self.get_content(norm_temp, temp=str(norm_temp))

    async def _get_temp_sensors(self) -> WidgetContent:
        d_temps = {}
        max_curr_temp = 0

        for fld, temp in self.get_temperatures().items():
            if fld in self.fields:
                max_curr_temp = max(max_curr_temp, temp)
                d_temps[fld] = temp

        await anyio.lowlevel.checkpoint()  # type: ignore

        norm_temp = min(max_curr_temp, self.max_temp)

        return self.get_content(norm_temp, **d_temps)

    async def run(self):
        while await self.sleep_interval():
            content = await self._coro_get_content()
            self.set_content_i(content)
