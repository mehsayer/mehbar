from pathlib import Path
import os

import psutil

from mehbar.widgets import Widget
from mehbar.exceptions import BarConfigError

class WidgetFanSpeed(Widget):
    def __init__(
        self,
        source: str | Path | None,
        max_speed: int,
        interval: int,
        label_format: str,
        ramp: list[str] | None = None,
    ):
        super().__init__(interval, label_format, ramp)

        self.path_fan = None

        expect_fields = ["ramp"]
        if source is not None:
            if isinstance(source, str):
                source_path = Path(source)
                if source_path.is_absolute():
                    self.path_fan = source_path
            elif isinstance(source, Path):
                self.path_fan = source


        if self.path_fan is None:
            expect_fields.extend(self.get_speeds().keys())
        elif not self.path_fan.is_file():
            raise BarConfigError(f"fan speed source file does not exist: {self.path_fan}")
        else:
            expect_fields.append("rpm")

        self.fields = []

        for fld in set(self.formatter.get_fields(label_format)):
            if fld not in expect_fields:
                raise BarConfigError(f"unknown label field: {fld}")
            else:
                self.fields.append(fld)

        self.max_speed = max(min(max_speed, 8000), 0)
        self.ramps = []

        for speed in range(max_speed + 1):
            ramp_val = str()

            if ramp is not None and (nramp := len(ramp)) > 0:
                ramp_idx = int(min(speed, max_speed - 1) / (max_speed / nramp))
                ramp_val = ramp[ramp_idx]

            self.ramps.append(ramp_val)

    def get_speeds(self) -> dict[str, int]:
        d_speeds = {}
        for name, l_sfan in psutil.sensors_fans().items():
            for sfan in l_sfan:
                selector = name
                if sfan.label:
                    selector += '/' + sfan.label

                d_speeds[selector] = round(sfan.current)
        return d_speeds


    def update(self):

        if self.path_fan:

            speed = 0
            with open(self.path_fan, "r", encoding="ascii") as fhandle:
                speed = int(fhandle.readline())

            norm_speed=min(speed, self.max_speed)
            self.format_label_idle(ramp=self.ramps[norm_speed],
                                   rpm=norm_speed)

        else:
            d_speeds = {}
            max_curr_speed = 0

            for fld, speed in self.get_speeds().items():
                if fld in self.fields:
                    max_curr_speed = max(max_curr_speed, speed)
                    d_speeds[fld] = speed

            norm_speed=min(max_curr_speed, self.max_speed)
            self.format_label_idle(ramp=self.ramps[norm_speed], **d_speeds)
