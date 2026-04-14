from pathlib import Path
import os

import psutil

from mehbar.widgets import Widget
from mehbar.exceptions import BarConfigError

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

        self.path_term = None

        zone_num = 0

        expect_fields = ["ramp"]
        if source is not None:
            try:
                zone_num = int(source)
            except ValueError:
                if isinstance(source, str):
                    source_path = Path(source)
                    if source_path.is_absolute():
                        self.path_term = source_path
                elif isinstance(source, Path):
                    self.path_term = source
            else:
                self.path_term = Path(f"/sys/class/thermal/thermal_zone{zone_num}/temp")

        if self.path_term is None:
            expect_fields.extend(self.get_temperatures().keys())
        elif not self.path_term.is_file():
            raise BarConfigError(f"termperature source file does not exist: {self.path_term}")
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

    def get_temperatures(self) -> dict[str, int]:
        d_temps = {}
        for name, l_swhtemp in psutil.sensors_temperatures().items():
            for swhtemp in l_swhtemp:
                selector = name
                if swhtemp.label:
                    selector += '/' + swhtemp.label

                d_temps[selector] = round(swhtemp.current)
        return d_temps


    def update(self):

        if self.path_term:

            temp = 0
            with open(self.path_term, "r", encoding="ascii") as fhandle:
                temp = int(fhandle.readline()) // 1000

            norm_temp=min(temp, self.max_temp)
            self.format_label_idle(ramp=self.ramps[norm_temp],
                                   temp=norm_temp)

        else:
            d_temps = {}
            max_curr_temp = 0

            for fld, temp in self.get_temperatures().items():
                if fld in self.fields:
                    max_curr_temp = max(max_curr_temp, temp)
                    d_temps[fld] = temp

            norm_temp=min(max_curr_temp, self.max_temp)
            self.format_label_idle(ramp=self.ramps[norm_temp], **d_temps)
