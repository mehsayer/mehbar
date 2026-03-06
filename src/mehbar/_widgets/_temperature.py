from pathlib import Path
from mehbar.widgets import BarWidget


class BarWidgetTemperature(BarWidget):
    def __init__(
        self,
        zone: int,
        max_temp: int,
        interval: int,
        label_format: str,
        ramp: list[str] | None = None,
    ):
        super().__init__(interval, label_format, ramp)

        self.path_term = Path(f"/sys/class/thermal/thermal_zone{zone}/temp")
        self.max_temp = max(min(max_temp, 200), 0)
        self.results = []

        # pre-generate formatted strings for all possible temperatures.
        # this uses O(max_temp) memory for O(1) update speed.
        # ranges of possible values in widgets are raltively narrow and do not
        # exceed at most a copule of hundreds of short strings, hence the
        # memory overhead is negligible.
        for temp in range(max_temp + 1):
            ramp_val = str()

            if ramp is not None and (nramp := len(ramp)) > 0:
                ramp_idx = int(min(temp, max_temp - 1) / (max_temp / nramp))
                ramp_val = ramp[ramp_idx]
            self.results.append(
                self.vformat_label(celsius=temp,
                                   fahrenheit=(temp * 1.8) + 32,
                                   ramp=ramp_val)
            )

    def update(self):
        temp = 0
        with open(self.path_term, "r", encoding="ascii") as fhandle:
            temp = int(fhandle.readline()) // 1000

        if self._last_value != temp:
            self._last_value = temp
            self.set_label_idle(self.results[min(temp, self.max_temp)])
