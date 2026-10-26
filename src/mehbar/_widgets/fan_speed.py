from pathlib import Path

from ._sensors import SensorWidget, read_int, sensor_values


class WidgetFanSpeed(SensorWidget):
    """Shows fan speeds in RPM. Without `path`, the `rpm` field is the speed
    of the fastest fan."""

    TYPE = "fan_speed"

    VALUE_FIELD = "rpm"
    MAX_OPTION = "max_speed"
    DEFAULT_MAX = 5000
    MAX_LIMIT = 8000

    def read_sensors(self) -> dict[str, int]:
        from psutil import sensors_fans

        return sensor_values(sensors_fans())

    def read_path(self, path: Path) -> int:
        return read_int(path)
