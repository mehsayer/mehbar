import enum
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anyio
from gi.repository import GLib  # type: ignore

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase

from ._dbus_facade import DBusFacade, is_null_path


class WiredOptions(enum.Flag):
    NONE = 0
    HWADDR = enum.auto()
    NAME = enum.auto()
    IPV4 = enum.auto()
    IPV6 = enum.auto()

    @classmethod
    def for_name(cls, name: str) -> "WiredOptions":
        return cls.__members__.get(name.upper(), cls.NONE)


@dataclass
class WiredInfo:
    iface: str | None
    name: str | None
    powered: bool
    connected: bool
    hwaddr: str | None = field(default=None)
    ipv4: str | None = field(default=None)
    ipv6: str | None = field(default=None)


def fill_missing(info: WiredInfo, options: WiredOptions) -> WiredInfo:
    from psutil import net_if_addrs

    get_hwaddr = WiredOptions.HWADDR in options and info.hwaddr is None
    get_ipv4 = WiredOptions.IPV4 in options and info.ipv4 is None
    get_ipv6 = WiredOptions.IPV6 in options and info.ipv6 is None

    if info.iface is not None and (get_hwaddr or get_ipv4 or get_ipv6):
        for snic in net_if_addrs().get(info.iface, []):
            if get_ipv4 and snic.family.name == "AF_INET":
                info.ipv4 = snic.address
            elif get_ipv6 and snic.family.name == "AF_INET6":
                if snic.address:
                    # strip the zone index of link-local addresses
                    info.ipv6 = snic.address.split("%", 1)[0]
            elif get_hwaddr and snic.family.name in ["AF_LINK", "AF_PACKET"]:
                if snic.address is not None:
                    info.hwaddr = snic.address.upper()

    return info


class WiredInfoQuery:
    """Backends are synchronous, call `get_info()` in a worker thread."""

    def get_info_(self, iface: str, options: WiredOptions) -> WiredInfo:
        raise NotImplementedError()

    def get_info(self, iface: str, options: WiredOptions) -> WiredInfo:
        info = self.get_info_(iface, options)
        return fill_missing(info, options) if info.connected else info


class NetworkManagerBackend(DBusFacade, WiredInfoQuery):
    BASE_SVC = "org.freedesktop.NetworkManager"
    BASE_OBJ = "/org/freedesktop/NetworkManager"
    BASE_IFACE = "org.freedesktop.NetworkManager"

    ACT_CONN_IFACE = "org.freedesktop.NetworkManager.Connection.Active"
    DEV_IFACE = "org.freedesktop.NetworkManager.Device"

    # NMDeviceState
    STATE_DISCONNECTED = 30
    STATE_ACTIVATED = 100

    def get_ipaddr(self, cfg_obj: str | None, ip_ver: int) -> str | None:
        if is_null_path(cfg_obj):
            return None

        addr_data = self.get_prop(
            cfg_obj, f"{self.BASE_IFACE}.IP{ip_ver}Config", "AddressData"
        )
        return addr_data[0]["address"] if addr_data else None

    def get_info_(self, iface: str, options: WiredOptions) -> WiredInfo:
        (dev_obj,) = self.call(
            self.BASE_OBJ,
            self.BASE_IFACE,
            "GetDeviceByIpIface",
            GLib.Variant("(s)", (iface,)),
            "(o)",
        )

        dev_props = self.get_all_props(dev_obj, self.DEV_IFACE)
        state = dev_props.get("State", 0)

        info = WiredInfo(
            iface,
            None,
            state >= self.STATE_DISCONNECTED,
            state == self.STATE_ACTIVATED,
        )

        if not info.connected:
            return info

        if WiredOptions.HWADDR in options:
            info.hwaddr = dev_props.get("HwAddress")

        if WiredOptions.IPV4 in options:
            info.ipv4 = self.get_ipaddr(dev_props.get("Ip4Config"), 4)

        if WiredOptions.IPV6 in options:
            info.ipv6 = self.get_ipaddr(dev_props.get("Ip6Config"), 6)

        if WiredOptions.NAME in options:
            act_conn_obj = dev_props.get("ActiveConnection")
            if not is_null_path(act_conn_obj):
                info.name = self.get_prop(act_conn_obj, self.ACT_CONN_IFACE, "Id")

        return info


class UnmanagedBackend(WiredInfoQuery):
    PATH_NET = Path("/sys/class/net")

    @staticmethod
    def _read(path: Path) -> str | None:
        try:
            return path.read_text().strip()
        except OSError:  # e.g. 'carrier' cannot be read while the link is down
            return None

    def get_info_(self, iface: str, options: WiredOptions) -> WiredInfo:
        base_path = self.PATH_NET / iface

        if not base_path.is_dir():
            raise FileNotFoundError(f"no such network interface: {iface}")

        pwrd = self._read(base_path / "operstate") == "up"
        connd = self._read(base_path / "carrier") == "1"

        return WiredInfo(iface, None, pwrd, pwrd and connd)


class ConnManBackend(DBusFacade, WiredInfoQuery):
    BASE_SVC = "net.connman"
    BASE_OBJ = "/"
    BASE_IFACE = "net.connman.Manager"

    TECH_OBJ = "/net/connman/technology/ethernet"
    TECH_IFACE = "net.connman.Technology"

    def get_info_(self, iface: str, options: WiredOptions) -> WiredInfo:
        (tech_props,) = self.call(
            self.TECH_OBJ, self.TECH_IFACE, "GetProperties", None, "(a{sv})"
        )

        info = WiredInfo(
            iface,
            None,
            tech_props.get("Powered", False),
            tech_props.get("Connected", False),
        )

        if not info.powered:
            return info

        (services,) = self.call(
            self.BASE_OBJ, self.BASE_IFACE, "GetServices", None, "(a(oa{sv}))"
        )

        info.connected = False

        for _, svc in services:
            if svc.get("Type") != "ethernet" or svc.get("State") not in (
                "ready",
                "online",
            ):
                continue

            if not (eth_obj := svc.get("Ethernet")) or eth_obj.get("Interface") != iface:
                continue

            info.connected = True

            if WiredOptions.NAME in options:
                info.name = svc.get("Name")

            if WiredOptions.HWADDR in options:
                info.hwaddr = eth_obj.get("Address")

            if WiredOptions.IPV4 in options:
                if ipv4_obj := svc.get("IPv4"):
                    info.ipv4 = ipv4_obj.get("Address")

            if WiredOptions.IPV6 in options:
                if ipv6_obj := svc.get("IPv6"):
                    info.ipv6 = ipv6_obj.get("Address")
            break

        return info


class WidgetWired(WidgetBase):
    """Shows the state of a wired network interface. Ramp entries are for
    the state: down, no link, connected."""

    TYPE = "wired"
    UNIQUE = False

    BACKEND_MAP = {
        "NetworkManager": NetworkManagerBackend,
        "connman": ConnManBackend,
        "unmanaged": UnmanagedBackend,
    }

    FMT_FIELDS = ["ipv4", "ipv6", "hwaddr", "name", "iface", "ramp"]

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        backend = self.cfg.get("backend", "unmanaged")

        if backend not in self.BACKEND_MAP:
            backends = ", ".join(repr(b) for b in self.BACKEND_MAP)
            raise BarConfigError(f"unknown backend '{backend}', not one of: {backends}")

        self.backend = self.BACKEND_MAP[backend]()

        if not isinstance(iface := self.cfg.get("iface"), str) or not iface:
            raise BarConfigError("'iface' must be specified")

        self.iface = iface

        self.qry_options = WiredOptions.NONE

        for fld in set(self.formatter.get_fields(self.content.label)):
            if fld not in self.FMT_FIELDS:
                raise BarConfigError(f"unknown label field: {fld}")
            self.qry_options |= WiredOptions.for_name(fld)

    def ramp_index(self, ramp_level: int) -> int | None:
        if ramp_level < 0 or not self.ramp:
            return None
        return min(ramp_level, len(self.ramp) - 1)

    async def run(self):
        last_info = None

        while await self.sleep_interval():
            info = await anyio.to_thread.run_sync(
                self.backend.get_info, self.iface, self.qry_options
            )

            if info != last_info:
                last_info = info

                if info.connected:
                    ramp_level = 2
                elif info.powered:
                    ramp_level = 1
                else:
                    ramp_level = 0

                self.set_new_content_i(ramp_level, **asdict(info))
