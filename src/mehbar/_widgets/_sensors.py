from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import anyio

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


def sensor_values(readings: Mapping[str, list[Any]]) -> dict[str, int]:
    """Maps psutil sensor readings to `<chip>/<label>` (or `<chip>`, if the
    sensor has no label) -> current value. Chips of the same name, e.g. two
    NVMe drives, are merged by psutil: repeated selectors get a `#<n>`
    suffix, `nvme/Composite`, `nvme/Composite#2`, in the order of hwmon
    devices."""
    values: dict[str, int] = {}

    for chip, entries in readings.items():
        for entry in entries:
            selector = f"{chip}/{entry.label}" if entry.label else chip
            unique = selector
            num = 1

            while unique in values:
                num += 1
                unique = f"{selector}#{num}"

            values[unique] = round(entry.current)

    return values


def check_fields(
    fields: Iterable[str], expected: Iterable[str], extra: Iterable[str] = ("ramp",)
) -> list[str]:
    """Raises `BarConfigError` for unknown fields, returns the expected ones."""
    expected = set(expected)
    allowed = expected | set(extra)

    if unknown := set(fields) - allowed:
        unk_s = ", ".join(sorted(repr(fld) for fld in unknown))
        exp_s = ", ".join(sorted(repr(fld) for fld in expected))
        raise BarConfigError(
            f"unknown fields: {unk_s}; expected one or more of: {exp_s}"
        )

    return [fld for fld in set(fields) if fld in expected]


def get_source_path(source: Any) -> Path | None:
    # TOML does not have a proper 'null' value
    if not isinstance(source, str) or source.lower() in ("", "none", "null"):
        return None

    path = Path(source)

    if not path.is_file():
        raise BarConfigError(f"source file does not exist: {path}")

    return path


def read_int(path: Path) -> int:
    with open(path, "r", encoding="ascii") as fhandle:
        return int(fhandle.readline())


class SensorWidget(WidgetBase):
    """Shows readings of hardware sensors, as `<chip>/<label>` fields, or of
    the `path` file. The `VALUE_FIELD` field is the reading of the file, or
    `auto_value()` of the sensors. The ramp follows the highest value shown,
    or of all sensors if none is. The widget is hidden while the sensors
    cannot be read."""

    UNIQUE = False
    POLLING = True

    VALUE_FIELD: str
    # the option setting the value of the last ramp entry
    MAX_OPTION: str
    DEFAULT_MAX: int
    MAX_LIMIT: int

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        max_value = self.cfg.get(self.MAX_OPTION, self.DEFAULT_MAX)

        if not isinstance(max_value, (int, float)):
            raise BarConfigError(f"'{self.MAX_OPTION}' must be a number")

        self.max_value = max(min(max_value, self.MAX_LIMIT), 1)
        self.max_ramp_level = self.cfg.get("max_ramp_level", self.max_value)

        self.path = get_source_path(self.cfg.get("path"))

        label_fields = self.label_fields()

        if self.path is None:
            expected = [*self.read_sensors(), self.VALUE_FIELD]
        else:
            expected = [self.VALUE_FIELD]

        self.fields = check_fields(label_fields, expected)

    def read_sensors(self) -> dict[str, int]:
        raise NotImplementedError()

    def read_path(self, path: Path) -> int:
        raise NotImplementedError()

    @classmethod
    def auto_value(cls, values: dict[str, int]) -> int | None:
        return max(values.values(), default=None)

    def _read(self) -> tuple[int, dict[str, int]]:
        if self.path is not None:
            value = self.read_path(self.path)
            return value, {self.VALUE_FIELD: value}

        values = self.read_sensors()
        shown = {fld: values[fld] for fld in self.fields if fld in values}

        if self.VALUE_FIELD in self.fields:
            if (value := self.auto_value(values)) is not None:
                shown[self.VALUE_FIELD] = value

        return max((shown or values).values(), default=0), shown

    async def run(self):
        seen = False

        while await self.sleep_interval():
            try:
                # reading all sensors takes a while
                highest, values = await anyio.to_thread.run_sync(self._read)
            except (OSError, ValueError):
                # a wrong path disables the widget
                if not seen:
                    raise
                # e.g. the sensor driver is reloaded
                self.set_available_i(False)
                continue

            seen = True
            self.set_available_i(True)
            self.set_new_content_i(min(max(highest, 0), self.max_value), **values)
