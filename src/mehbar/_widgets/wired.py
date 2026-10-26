from dataclasses import dataclass, field

from ._dbus_facade import is_null_path
from ._net import (
    PATH_NET,
    ConnManMixin,
    NetOptions,
    NetworkManagerMixin,
    PolledNetWidget,
    ethernet_interfaces,
    fill_addresses,
    read_attr,
)


@dataclass
class WiredInfo:
    iface: str | None
    name: str | None
    powered: bool
    connected: bool
    hwaddr: str | None = field(default=None)
    ipv4: str | None = field(default=None)
    ipv6: str | None = field(default=None)


class WiredInfoQuery:
    """Backends are synchronous, call `get_info()` in a worker thread."""

    def get_info_(self, iface: str, options: NetOptions) -> WiredInfo:
        raise NotImplementedError()

    def get_info(self, iface: str, options: NetOptions) -> WiredInfo:
        info = self.get_info_(iface, options)
        return fill_addresses(info, options) if info.connected else info


class NetworkManagerBackend(NetworkManagerMixin, WiredInfoQuery):
    # NMDeviceState: unavailable is managed, but without a link
    STATE_UNAVAILABLE = 20
    STATE_ACTIVATED = 100

    def get_info_(self, iface: str, options: NetOptions) -> WiredInfo:
        _, dev_props = self.get_device(iface)
        state = dev_props.get("State", 0)

        info = WiredInfo(
            iface,
            None,
            state >= self.STATE_UNAVAILABLE,
            state == self.STATE_ACTIVATED,
        )

        if not info.connected:
            return info

        self.fill_device(info, dev_props, options)

        if NetOptions.NAME in options:
            act_conn_obj = dev_props.get("ActiveConnection")
            if not is_null_path(act_conn_obj):
                info.name = self.get_prop(act_conn_obj, self.ACT_CONN_IFACE, "Id")

        return info


class UnmanagedBackend(WiredInfoQuery):
    PATH_NET = PATH_NET
    IFF_UP = 0x1

    def get_info_(self, iface: str, options: NetOptions) -> WiredInfo:
        base_path = self.PATH_NET / iface

        if not base_path.is_dir():
            # e.g. an USB adapter is unplugged
            return WiredInfo(iface, None, False, False)

        # the operational state is 'down' without a link, it does not tell
        # whether the interface is up
        try:
            is_up = bool(int(read_attr(base_path / "flags") or "0", 16) & self.IFF_UP)
        except ValueError:
            is_up = False

        # 'unknown' if the driver does not report the state, 'dormant' while
        # authenticating, e.g. with 802.1X
        connected = (
            is_up
            and read_attr(base_path / "carrier") == "1"
            and read_attr(base_path / "operstate") in ("up", "unknown")
        )

        return WiredInfo(iface, None, is_up, connected)


class ConnManBackend(ConnManMixin, WiredInfoQuery):
    TECH_OBJ = "/net/connman/technology/ethernet"
    TECH_IFACE = "net.connman.Technology"

    def get_info_(self, iface: str, options: NetOptions) -> WiredInfo:
        (tech_props,) = self.call(
            self.TECH_OBJ, self.TECH_IFACE, "GetProperties", None, "(a{sv})"
        )

        info = WiredInfo(iface, None, tech_props.get("Powered", False), False)

        if not info.powered or (svc := self.get_service(iface, "ethernet")) is None:
            return info

        info.connected = True
        self.fill_service(info, svc, options)

        if NetOptions.NAME in options:
            info.name = svc.get("Name")

        return info


class WidgetWired(PolledNetWidget):
    """Shows the state of a wired network interface. Ramp entries are for
    the state: down, no link, connected."""

    TYPE = "wired"
    RAMP_BY_STATE = True

    BACKEND_MAP = {
        "NetworkManager": NetworkManagerBackend,
        "connman": ConnManBackend,
        "unmanaged": UnmanagedBackend,
    }

    BACKEND_SERVICES = {
        "org.freedesktop.NetworkManager": "NetworkManager",
        "net.connman": "connman",
    }

    FMT_FIELDS = ["ipv4", "ipv6", "hwaddr", "name", "iface", "ramp"]

    def detect_iface(self) -> str | None:
        return next(iter(ethernet_interfaces()), None)

    def empty_info(self, iface: str) -> WiredInfo:
        return WiredInfo(iface, None, False, False)

    def info_ramp_level(self, info: WiredInfo) -> int:
        if info.connected:
            return 2
        return 1 if info.powered else 0
