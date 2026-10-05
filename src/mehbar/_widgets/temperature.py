from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase

from ._sensors import check_fields, get_source_path, read_int, sensor_values


def get_temperatures() -> dict[str, int]:
    from psutil import sensors_temperatures

    return sensor_values(sensors_temperatures())


class WidgetTemperature(WidgetBase):
    """Shows temperatures read either from the `path` file, as the `temp`
    field, or from hardware sensors, as `<chip>/<label>` fields. The ramp
    follows the highest temperature shown, or of all sensors if none is."""

    UNIQUE = False

    TYPE = "temperature"

    MAX_TEMP_LO = 100
    MAX_TEMP_HI = 150

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.max_temp = min(
            self.cfg.get("max_temp", self.MAX_TEMP_LO), self.MAX_TEMP_HI
        )

        self.max_ramp_level = self.cfg.get("max_ramp_level", self.max_temp)

        self.path = get_source_path(self.cfg.get("path"))

        label_fields = self.formatter.get_fields(self.content.label)

        if self.path is None:
            self.fields = check_fields(label_fields, get_temperatures())
        else:
            self.fields = check_fields(label_fields, ["temp"])

    def _read(self) -> tuple[int, dict[str, int]]:
        if self.path is not None:
            temp = read_int(self.path) // 1000
            return temp, {"temp": temp}

        temps = get_temperatures()
        shown = {fld: temps[fld] for fld in self.fields if fld in temps}

        return max((shown or temps).values(), default=0), shown

    async def run(self):
        while await self.sleep_interval():
            hottest, values = self._read()
            self.set_new_content_i(min(max(hottest, 0), self.max_temp), **values)
