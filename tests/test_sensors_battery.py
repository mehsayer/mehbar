from collections import namedtuple

import pytest

from mehbar._widgets._sensors import check_fields, sensor_values
from mehbar._widgets.battery import read_batteries
from mehbar.exceptions import BarConfigError

Reading = namedtuple("Reading", "label current")


def test_sensor_values_keeps_sensors_of_same_named_chips():
    readings = {
        # psutil merges chips of the same name, e.g. two NVMe drives
        "nvme": [Reading("Composite", 41.8), Reading("Composite", 36.2)],
        "k10temp": [Reading("Tctl", 30.4)],
        "acpitz": [Reading("", 27.0)],
    }

    assert sensor_values(readings) == {
        "nvme/Composite": 42,
        "nvme/Composite#2": 36,
        "k10temp/Tctl": 30,
        "acpitz": 27,
    }


def test_cpu_temperature_is_preferred():
    from mehbar._widgets.temperature import WidgetTemperature

    auto_value = WidgetTemperature.auto_value
    values = {"nvme/Composite": 45, "k10temp/Tctl": 38, "acpitz": 27}

    assert auto_value(values) == 38
    # the hottest sensor, if the CPU sensor is not known
    assert auto_value({"nvme/Composite": 45, "foo/bar": 50}) == 50
    assert auto_value({}) is None


def test_check_fields():
    assert check_fields(["temp", "ramp"], ["temp"]) == ["temp"]

    with pytest.raises(BarConfigError, match="'bogus'"):
        check_fields(["bogus"], ["temp"])


def _supply(root, name, **attrs):
    path = root / name
    path.mkdir()
    for attr, value in attrs.items():
        (path / attr).write_text(f"{value}\n")


def test_no_batteries(tmp_path):
    _supply(tmp_path, "AC", type="Mains", online=1)
    assert read_batteries(tmp_path) is None


def test_two_batteries_are_combined(tmp_path):
    _supply(tmp_path, "AC", type="Mains", online=0)
    common = {"type": "Battery", "status": "Discharging", "energy_full": 50_000_000}
    _supply(tmp_path, "BAT0", **common, energy_now=10_000_000, power_now=5_000_000)
    _supply(tmp_path, "BAT1", **common, energy_now=40_000_000, power_now=5_000_000)

    info = read_batteries(tmp_path)

    assert info.percent == 50
    assert info.status == "discharging"
    assert not info.plugged
    # 50 Wh left at 10 W
    assert info.secsleft == 5 * 3600


def test_peripheral_batteries_are_ignored(tmp_path):
    _supply(tmp_path, "BAT0", type="Battery", status="Full", capacity=100)
    _supply(tmp_path, "hidpp_battery_0", type="Battery", scope="Device", capacity=5)

    assert read_batteries(tmp_path).percent == 100


@pytest.mark.parametrize(("online", "plugged"), [(0, False), (1, True), (2, True)])
def test_usb_pd_adapters_are_plugged(tmp_path, online, plugged):
    # 2 is 'online programmable', e.g. a USB PD charger
    _supply(tmp_path, "ucsi-source-psy-USBC000:001", type="USB", online=online)
    _supply(tmp_path, "BAT0", type="Battery", status="Not charging", capacity=80)

    assert read_batteries(tmp_path).plugged is plugged
