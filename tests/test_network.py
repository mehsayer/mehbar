import pytest
from gi.repository import Gio, GLib

from mehbar._widgets import _wifi, wired
from mehbar._widgets._dbus_facade import DBusFacade, DBusUnavailable
from mehbar._widgets._wifi import WifiOptions, rssi_to_strength
from mehbar._widgets.network_rate import WidgetNetworkRate
from mehbar._widgets.wired import WiredOptions


@pytest.mark.parametrize(
    ("rssi", "percent"), [(-30, 100), (-20, 100), (-65, 50), (-100, 0), (-110, 0)]
)
def test_rssi_to_strength(rssi, percent):
    assert rssi_to_strength(rssi) == percent


WIRELESS = """\
Inter-| sta-|   Quality        |   Discarded packets               | Missed | WE
 face | tus | link level noise |  nwid  crypt   frag  retry   misc | beacon | 22
wlan0: 0000   45.  -65.  -256        0      0      0      0      0        0
wlan1: 0000    0.    0.  -256        0      0      0      0      0        0
"""


@pytest.mark.parametrize(("iface", "rssi"), [("wlan0", -65), ("wlan1", None)])
def test_unmanaged_wifi_signal(tmp_path, monkeypatch, iface, rssi):
    # the level is 0 while disconnected, which is not a signal of 0 dBm
    (tmp_path / "wireless").write_text(WIRELESS)
    monkeypatch.setattr(_wifi.UnmanagedBackend, "PATH_WIRELESS", tmp_path / "wireless")

    info = _wifi.UnmanagedBackend().get_info(iface, WifiOptions.SIGNAL)

    assert info.rssi == rssi
    assert info.percentage == (None if rssi is None else rssi_to_strength(rssi))


def test_unmanaged_wired_missing_interface_is_down(tmp_path, monkeypatch):
    monkeypatch.setattr(wired.UnmanagedBackend, "PATH_NET", tmp_path)

    info = wired.UnmanagedBackend().get_info("enx0", WiredOptions.NONE)

    assert not info.powered and not info.connected


class _Rate:
    _conv_rate = WidgetNetworkRate._conv_rate

    def __init__(self, conv_map):
        self.conv_map = sorted(conv_map.items(), key=lambda kv: kv[1], reverse=True)


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (0, (0, "B/s")),
        (500, (500, "B/s")),
        (2048, (2, "KiB/s")),
        (3 << 20, (3, "MiB/s")),
    ],
)
def test_network_rate_units(rate, expected):
    assert _Rate(WidgetNetworkRate.DEFAULT_CONVERSIONS)._conv_rate(rate) == expected


def test_network_rate_below_the_smallest_unit():
    # 500 B/s is not 500 KiB/s
    assert _Rate({"KiB/s": 1024, "MiB/s": 1 << 20})._conv_rate(500) == (0, "KiB/s")


@pytest.fixture
def system_bus():
    try:
        return Gio.bus_get_sync(Gio.BusType.SYSTEM, None)
    except GLib.Error:
        pytest.skip("no system bus")


def test_missing_service_is_unavailable(system_bus):
    facade = DBusFacade(system_bus, "org.example.MehbarTest")

    with pytest.raises(DBusUnavailable, match="org.example.MehbarTest"):
        facade.call("/", "org.example.MehbarTest", "Method")


def test_missing_object_is_unavailable(system_bus):
    with pytest.raises(DBusUnavailable):
        DBusFacade(system_bus).get_all_props("/org/example/none", "org.example.None")


def test_other_errors_are_raised_as_is(system_bus):
    with pytest.raises(GLib.Error) as exc_info:
        DBusFacade(system_bus).call(
            "/org/freedesktop/DBus", "org.freedesktop.DBus", "NoSuchMethod"
        )
    assert not isinstance(exc_info.value, DBusUnavailable)


def test_bus_does_not_exit_on_close(system_bus):
    facade = DBusFacade()
    assert facade.bus.get_exit_on_close() is False
