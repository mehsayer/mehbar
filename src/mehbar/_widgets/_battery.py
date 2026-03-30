from mehbar.widgets import BarWidget
from mehbar.exceptions import CapabilityError
import psutil
from itertools import batched
from mehbar.tools import FormattableTimeDelta


class BarWidgetBattery(BarWidget):

    MAX_CHARGE=100

    def __init__(self, interval: int, label_format: str, ramp: list[str] | None = None):
        super().__init__(interval, label_format, ramp)

        self.ramps = []

        if ramp is not None and ramp:
            # TODO: raise exception if ramp too short, comment on expected format or just throw exception if no battery

            self.ramps.extend((ramp[0], ramp[0]))

            lexpand = []

            if len(ramp) - 1 % 2 == 1:
                lexpand.append("?")

            ramp_batched = list(batched(ramp[1:] + lexpand, n=2))

            for charge in range(self.MAX_CHARGE + 1):
                if (nramp := len(ramp_batched)) > 0:
                    ramp_idx = int(min(charge, self.MAX_CHARGE - 1) / (self.MAX_CHARGE / nramp))
                    self.ramps.append(ramp_batched[ramp_idx])
        else:
            self.ramps.extend([(None, None)] * (self.MAX_CHARGE + 1))

    def update(self):

        if (bat_st := psutil.sensors_battery()) is not None:

            if bat_st != self._last_value:
                self._last_value = bat_st

                percent = min(int(bat_st.percent), self.MAX_CHARGE)

                timeleft = None

                if bat_st.secsleft not in [psutil.POWER_TIME_UNLIMITED, psutil.POWER_TIME_UNKNOWN]:
                    timeleft = FormattableTimeDelta(bat_st.secsleft)

                self.format_label_idle(ramp=self.ramps[percent + 2][bat_st.power_plugged],
                                       timeleft=timeleft,
                                       percent=percent)
        else:
            self.format_label_idle(ramp=self.ramps[0][0], timeleft=None, percent=0)
            # raise CapabilityError("no battery detected")
