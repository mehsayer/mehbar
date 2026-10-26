import pytest
from gi.repository import Gio, GLib

from mehbar._widgets import _net, _wifi, wired
from mehbar._widgets._bluetooth import BluezBackend
from mehbar._widgets._dbus_facade import DBusFacade, DBusUnavailable
from mehbar._widgets._net import NetOptions
from mehbar._widgets._wifi import rssi_to_strength, strength_to_rssi
from mehbar._widgets.network_rate import WidgetNetworkRate


@pytest.mark.parametrize(
    ("rssi", "percent"),
    # as NetworkManager maps them
    [(-30, 100), (-40, 100), (-65, 59), (-70, 50), (-100, 0), (-110, 0)],
)
def test_rssi_to_strength(rssi, percent):
    assert rssi_to_strength(rssi) == percent


@pytest.mark.parametrize("percent", [0, 25, 50, 75, 100])
def test_strength_to_rssi_is_the_inverse(percent):
    assert rssi_to_strength(strength_to_rssi(percent)) == percent


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

    info = _wifi.UnmanagedBackend().get_info(iface, NetOptions.SIGNAL)

    assert info.rssi == rssi
    assert info.percent == (None if rssi is None else rssi_to_strength(rssi))


def test_unmanaged_wired_missing_interface_is_down(tmp_path, monkeypatch):
    monkeypatch.setattr(wired.UnmanagedBackend, "PATH_NET", tmp_path)

    info = wired.UnmanagedBackend().get_info("enx0", NetOptions.NONE)

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


@pytest.mark.parametrize(
    ("flags", "carrier", "operstate", "powered", "connected"),
    [
        ("0x1003", "1", "up", True, True),
        # no cable: the state is 'down', but the interface is up
        ("0x1003", "0", "down", True, False),
        ("0x1002", None, "down", False, False),
        # drivers not reporting the state
        ("0x1003", "1", "unknown", True, True),
        # e.g. 802.1X authentication
        ("0x1003", "1", "dormant", True, False),
    ],
)
def test_unmanaged_wired_states(
    tmp_path, monkeypatch, flags, carrier, operstate, powered, connected
):
    iface = tmp_path / "eth0"
    iface.mkdir()
    (iface / "flags").write_text(flags + "\n")
    (iface / "operstate").write_text(operstate + "\n")
    if carrier is not None:
        (iface / "carrier").write_text(carrier + "\n")
    monkeypatch.setattr(wired.UnmanagedBackend, "PATH_NET", tmp_path)

    info = wired.UnmanagedBackend().get_info("eth0", NetOptions.NONE)

    assert (info.powered, info.connected) == (powered, connected)


def _iface(root, name, type_="1", device=True, wireless=False):
    path = root / name
    path.mkdir()
    (path / "type").write_text(type_ + "\n")
    if device:
        (path / "device").mkdir()
    if wireless:
        (path / "wireless").mkdir()


def test_interface_detection(tmp_path, monkeypatch):
    _iface(tmp_path, "lo", type_="772", device=False)
    _iface(tmp_path, "docker0", device=False)
    _iface(tmp_path, "enp3s0")
    _iface(tmp_path, "eno1")
    _iface(tmp_path, "wlan0", wireless=True)
    monkeypatch.setattr(_net, "PATH_NET", tmp_path)

    assert _net.ethernet_interfaces() == ["eno1", "enp3s0"]
    assert _net.wireless_interfaces() == ["wlan0"]


def test_backend_detection(monkeypatch):
    running = {"net.connman.iwd"}

    def _call(self, obj, iface, method, args=None, reply_type=None):
        return (args.unpack()[0] in running,)

    monkeypatch.setattr(DBusFacade, "call", _call)
    services = {
        "org.freedesktop.NetworkManager": "NetworkManager",
        "net.connman.iwd": "iwd",
    }

    assert _net.detect_backend(services, "unmanaged") == "iwd"

    running.clear()
    assert _net.detect_backend(services, "unmanaged") == "unmanaged"


@pytest.mark.parametrize(
    ("uuid", "volume", "percent"),
    [
        # A2DP
        ("0000110b-0000-1000-8000-00805f9b34fb", 127, 100),
        ("0000110B-0000-1000-8000-00805F9B34FB", 64, 50),
        # LE Audio
        ("00008f01-0000-1000-8000-00805f9b34fb", 255, 100),
        ("00008f01-0000-1000-8000-00805f9b34fb", 64, 25),
    ],
)
def test_bluetooth_volume_percent(uuid, volume, percent):
    assert BluezBackend.volume_percent({"UUID": uuid, "Volume": volume}) == percent
    assert BluezBackend.volume_percent({"UUID": uuid}) is None
