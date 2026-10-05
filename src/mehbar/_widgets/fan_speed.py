from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase

from ._sensors import check_fields, get_source_path, read_int, sensor_values


def get_speeds() -> dict[str, int]:
    from psutil import sensors_fans

    return sensor_values(sensors_fans())


class WidgetFanSpeed(WidgetBase):
    """Shows fan speeds read either from the `path` file, as the `rpm` field,
    or from hardware sensors, as `<chip>/<label>` fields. The ramp follows
    the highest speed shown, or of all fans if none is."""

    UNIQUE = False
    TYPE = "fan_speed"

    DEFAULT_MAX_SPEED = 5000
    MAX_SPEED = 8000

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        max_speed = self.cfg.get("max_speed", self.DEFAULT_MAX_SPEED)
        self.max_speed = max(min(max_speed, self.MAX_SPEED), 1)

        self.max_ramp_level = self.cfg.get("max_ramp_level", self.max_speed)

        self.path = get_source_path(self.cfg.get("path"))

        label_fields = self.formatter.get_fields(self.content.label)

        if self.path is None:
            self.fields = check_fields(label_fields, get_speeds())
        else:
            self.fields = check_fields(label_fields, ["rpm"])

    def _read(self) -> tuple[int, dict[str, int]]:
        if self.path is not None:
            rpm = read_int(self.path)
            return rpm, {"rpm": rpm}

        speeds = get_speeds()
        shown = {fld: speeds[fld] for fld in self.fields if fld in speeds}

        return max((shown or speeds).values(), default=0), shown

    async def run(self):
        while await self.sleep_interval():
            fastest, values = self._read()
            self.set_new_content_i(min(max(fastest, 0), self.max_speed), **values)
