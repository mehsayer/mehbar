import errno

from mehbar.resource_manager import ResourceManager
from mehbar.tools import FormattableTimeDelta
from mehbar.widget import WidgetBase, WidgetContent


class WidgetBattery(WidgetBase):
    MAX_CHARGE = 100
    TYPE = "battery"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)
        self._last_st = None

    def get_ramp(self, ramp_level: int = -1) -> WidgetContent | None:

        if ramp_level not in self.ramp_index_cache:
            ramp = self.cfg.get("ramp")
            content = None

            if ramp is not None and ramp:
                ramp_offset = 0
                ramp_level_ = 0

                if ramp_level >= 1000:
                    ramp_level_ = ramp_level % 1000
                    ramp_offset += 1

                if ramp_level_ >= 0 and ramp is not None and ramp:
                    level_ = max(min(ramp_level_, self.MAX_CHARGE - 1), 0)

                    idx = int(level_ / (self.MAX_CHARGE / (len(ramp) / 2)))
                    idx = idx * 2 + ramp_offset

                    content = WidgetContent.parse(ramp[idx])

            self.ramp_index_cache[ramp_level] = content
        return self.ramp_index_cache[ramp_level]

    async def run(self):

        from psutil import POWER_TIME_UNKNOWN, POWER_TIME_UNLIMITED, sensors_battery

        while await self.sleep_interval():
            if (bat_st := sensors_battery()) is not None:
                if bat_st != self._last_st:
                    self._last_st = bat_st

                    percent = max(min(int(bat_st.percent), self.MAX_CHARGE), 0)

                    timeleft = None

                    if bat_st.secsleft not in [
                        POWER_TIME_UNLIMITED,
                        POWER_TIME_UNKNOWN,
                    ]:
                        timeleft = FormattableTimeDelta(bat_st.secsleft)

                    ramp_level = percent + (int(bat_st.power_plugged) * 1000)

                    self.set_new_content_i(
                        ramp_level=ramp_level,
                        percent=percent,
                        timeleft=timeleft,
                    )
            else:
                raise OSError(errno.ENODEV, "no battery detected")
