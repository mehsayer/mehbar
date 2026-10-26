from pathlib import Path

from ._sensors import SensorWidget, read_int, sensor_values

# sensors of CPU packages, in order of preference
CPU_SENSORS = (
    "coretemp/Package id 0",
    "k10temp/Tctl",
    "zenpower/Tctl",
    "zenpower/Tdie",
    "cpu_thermal",
    "cpu-thermal",
    "acpitz",
)


class WidgetTemperature(SensorWidget):
    """Shows temperatures in °C. Without `path`, the `temp` field is the CPU
    temperature, or the highest one if the CPU sensor is not known."""

    TYPE = "temperature"
    DEFAULT_TOOLTIP = "Temperature: {temp}°C"

    VALUE_FIELD = "temp"
    MAX_OPTION = "max_temp"
    DEFAULT_MAX = 100
    MAX_LIMIT = 150

    def read_sensors(self) -> dict[str, int]:
        from psutil import sensors_temperatures

        return sensor_values(sensors_temperatures())

    def read_path(self, path: Path) -> int:
        # in millidegrees
        return round(read_int(path) / 1000)

    @classmethod
    def auto_value(cls, values: dict[str, int]) -> int | None:
        for sensor in CPU_SENSORS:
            if sensor in values:
                return values[sensor]
        return super().auto_value(values)
