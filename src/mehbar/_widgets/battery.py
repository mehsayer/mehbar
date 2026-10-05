import errno
from dataclasses import dataclass
from pathlib import Path

from mehbar.tools import FormattableTimeDelta
from mehbar.widget import WidgetBase

POWER_SUPPLY_PATH = Path("/sys/class/power_supply")


@dataclass(frozen=True)
class BatteryInfo:
    percent: int
    plugged: bool
    # charging, discharging, full, not charging or unknown
    status: str
    # until empty when discharging, until full when charging
    secsleft: int | None


def _read(path: Path, name: str) -> str | None:
    try:
        return (path / name).read_text().strip()
    except OSError:
        return None


def _read_int(path: Path, name: str) -> int | None:
    try:
        return int(_read(path, name))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class _Battery:
    status: str
    capacity: int | None
    # either energy (µWh, µW) or charge (µAh, µA)
    unit: str | None
    now: int | None
    full: int | None
    rate: int | None
    time_to_empty: int | None
    time_to_full: int | None

    @classmethod
    def read(cls, path: Path) -> "_Battery":
        unit, now, full, rate = None, None, None, None

        for unit_, rate_name in (("energy", "power_now"), ("charge", "current_now")):
            now_ = _read_int(path, f"{unit_}_now")
            full_ = _read_int(path, f"{unit_}_full")

            if now_ is not None and full_:
                unit, now, full = unit_, now_, full_
                rate = abs(_read_int(path, rate_name) or 0) or None
                break

        return cls(
            (_read(path, "status") or "unknown").lower(),
            _read_int(path, "capacity"),
            unit,
            now,
            full,
            rate,
            _read_int(path, "time_to_empty_now") or _read_int(path, "time_to_empty_avg"),
            _read_int(path, "time_to_full_now") or _read_int(path, "time_to_full_avg"),
        )


def read_batteries(root: Path = POWER_SUPPLY_PATH) -> BatteryInfo | None:
    """Combines all system batteries, `None` if there are none."""
    batteries: list[_Battery] = []
    adapters_online: list[bool] = []

    for path in sorted(root.iterdir()) if root.is_dir() else []:
        match _read(path, "type"):
            case "Battery":
                # skip batteries of peripherals, e.g. mice and headsets
                if _read(path, "scope") == "Device" or _read(path, "present") == "0":
                    continue
                batteries.append(_Battery.read(path))
            case "Mains" | "USB":
                adapters_online.append(_read(path, "online") == "1")

    if not batteries:
        return None

    statuses = {bat.status for bat in batteries}

    if "charging" in statuses:
        status = "charging"
    elif "discharging" in statuses:
        status = "discharging"
    elif statuses == {"full"}:
        status = "full"
    elif "not charging" in statuses:
        status = "not charging"
    else:
        status = "unknown"

    if adapters_online:
        plugged = any(adapters_online)
    else:
        plugged = status in ("charging", "full", "not charging")

    units = {bat.unit for bat in batteries}
    same_unit = len(units) == 1 and None not in units

    if same_unit:
        total_now = sum(bat.now or 0 for bat in batteries)
        total_full = sum(bat.full or 0 for bat in batteries)
        percent = round(100 * total_now / total_full)
    else:
        capacities = [bat.capacity for bat in batteries if bat.capacity is not None]
        percent = round(sum(capacities) / len(capacities)) if capacities else 0

    secsleft = None
    total_rate = sum(bat.rate or 0 for bat in batteries)

    if status == "discharging":
        if same_unit and total_rate:
            secsleft = round(3600 * total_now / total_rate)
        else:
            # batteries are usually drained one after another
            times = [
                bat.time_to_empty
                for bat in batteries
                if bat.status == "discharging" and bat.time_to_empty
            ]
            secsleft = sum(times) if times else None
    elif status == "charging":
        if same_unit and total_rate:
            secsleft = round(3600 * (total_full - total_now) / total_rate)
        else:
            times = [
                bat.time_to_full
                for bat in batteries
                if bat.status == "charging" and bat.time_to_full
            ]
            secsleft = max(times) if times else None

    return BatteryInfo(max(0, min(percent, 100)), plugged, status, secsleft)


class WidgetBattery(WidgetBase):
    """Shows the combined charge of all system batteries. Label fields:
    `percent`, `timeleft`, `status`, `plugged`. Ramp entries come in pairs:
    on battery, plugged in."""

    MAX_CHARGE = 100
    TYPE = "battery"

    def ramp_index(self, ramp_level: int) -> int | None:
        if ramp_level < 0 or not self.ramp:
            return None

        plugged, level = divmod(ramp_level, 1000)
        level = max(min(level, self.MAX_CHARGE - 1), 0)
        npairs = max(len(self.ramp) // 2, 1)

        idx = int(level / (self.MAX_CHARGE / npairs)) * 2 + (1 if plugged else 0)

        return min(idx, len(self.ramp) - 1)

    async def run(self):
        while await self.sleep_interval():
            if (info := read_batteries()) is None:
                raise OSError(errno.ENODEV, "no battery detected")

            timeleft = None

            if info.secsleft is not None and info.secsleft > 0:
                timeleft = FormattableTimeDelta(info.secsleft)

            self.set_new_content_i(
                info.percent + (1000 if info.plugged else 0),
                percent=info.percent,
                timeleft=timeleft,
                status=info.status,
                plugged=info.plugged,
            )
