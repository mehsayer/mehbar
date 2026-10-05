from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from mehbar.exceptions import BarConfigError


def sensor_values(readings: Mapping[str, list[Any]]) -> dict[str, int]:
    """Maps psutil sensor readings to `<chip>/<label>` (or `<chip>`, if the
    sensor has no label) -> current value."""
    values = {}

    for chip, entries in readings.items():
        for entry in entries:
            selector = f"{chip}/{entry.label}" if entry.label else chip
            values[selector] = round(entry.current)

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
