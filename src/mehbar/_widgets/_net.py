"""Parts shared by the wifi and wired widgets: label field options, network
interface and backend detection, NetworkManager and ConnMan helpers and the
polling widget."""

import enum
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anyio
from gi.repository import GLib  # type: ignore

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase

from ._dbus_facade import DBusFacade, DBusUnavailable, is_null_path

PATH_NET = Path("/sys/class/net")

# ARPHRD_ETHER, wireless interfaces have it too
IFACE_TYPE_ETHER = "1"


class NetOptions(enum.Flag):
    """Information to query, named after label fields."""

    NONE = 0
    HWADDR = enum.auto()
    IPV4 = enum.auto()
    IPV6 = enum.auto()
    NAME = enum.auto()
    PERCENT = enum.auto()
    RSSI = enum.auto()
    SECURITY = enum.auto()
    SSID = enum.auto()
    SIGNAL = PERCENT | RSSI

    @classmethod
    def for_name(cls, name: str) -> "NetOptions":
        return cls.__members__.get(name.upper(), cls.NONE)


def read_attr(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:  # e.g. 'carrier' cannot be read while the link is down
        return None


def _is_wireless(path: Path) -> bool:
    return (path / "wireless").is_dir() or (path / "phy80211").exists()


def _interfaces() -> list[Path]:
    try:
        return sorted(PATH_NET.iterdir())
    except OSError:
        return []


def wireless_interfaces() -> list[str]:
    return [path.name for path in _interfaces() if _is_wireless(path)]


def ethernet_interfaces() -> list[str]:
    """Physical, wired interfaces: virtual ones have no device."""
    return [
        path.name
        for path in _interfaces()
        if (path / "device").exists()
        and read_attr(path / "type") == IFACE_TYPE_ETHER
        and not _is_wireless(path)
    ]


def fill_addresses(info: Any, options: NetOptions) -> Any:
    """Fills in the `hwaddr`, `ipv4` and `ipv6` attributes of `info` that the
    backend did not provide, from the addresses of `info.iface`."""
    from psutil import net_if_addrs

    get_hwaddr = NetOptions.HWADDR in options and info.hwaddr is None
    get_ipv4 = NetOptions.IPV4 in options and info.ipv4 is None
    get_ipv6 = NetOptions.IPV6 in options and info.ipv6 is None

    if info.iface is not None and (get_hwaddr or get_ipv4 or get_ipv6):
        for snic in net_if_addrs().get(info.iface, []):
            # the first address of each family
            if get_ipv4 and snic.family.name == "AF_INET":
                info.ipv4 = snic.address
                get_ipv4 = False
            elif get_ipv6 and snic.family.name == "AF_INET6" and snic.address:
                # strip the zone index of link-local addresses
                info.ipv6 = snic.address.split("%", 1)[0]
                get_ipv6 = False
            elif get_hwaddr and snic.family.name in ("AF_LINK", "AF_PACKET"):
                if snic.address is not None:
                    info.hwaddr = snic.address.upper()
                    get_hwaddr = False

    return info


def detect_backend(services: dict[str, str], default: str) -> str:
    """Returns the backend of the first D-Bus service in `services`, which
    maps service names to backend names, that is running. Blocks, call it
    in a worker thread."""
    bus = DBusFacade()

    for svc, backend in services.items():
        try:
            (running,) = bus.call(
                "/org/freedesktop/DBus",
                "org.freedesktop.DBus",
                "NameHasOwner",
                GLib.Variant("(s)", (svc,)),
                "(b)",
            )
        except (DBusUnavailable, GLib.Error):
            break

        if running:
            return backend

    return default


class NetworkManagerMixin(DBusFacade):
    BASE_SVC = "org.freedesktop.NetworkManager"
    BASE_OBJ = "/org/freedesktop/NetworkManager"
    BASE_IFACE = "org.freedesktop.NetworkManager"

    DEV_IFACE = "org.freedesktop.NetworkManager.Device"
    ACT_CONN_IFACE = "org.freedesktop.NetworkManager.Connection.Active"

    UNAVAILABLE_ERROR_NAMES = frozenset(
        {"org.freedesktop.NetworkManager.UnknownDevice"}
    )

    def get_device(self, iface: str) -> tuple[str, dict[str, Any]]:
        """Returns the object path and properties of the device."""
        (dev_obj,) = self.call(
            self.BASE_OBJ,
            self.BASE_IFACE,
            "GetDeviceByIpIface",
            GLib.Variant("(s)", (iface,)),
            "(o)",
        )
        return dev_obj, self.get_all_props(dev_obj, self.DEV_IFACE)

    def get_ipaddr(self, cfg_obj: str | None, ip_ver: int) -> str | None:
        if is_null_path(cfg_obj):
            return None

        addr_data = self.get_prop(
            cfg_obj, f"{self.BASE_IFACE}.IP{ip_ver}Config", "AddressData"
        )
        return addr_data[0]["address"] if addr_data else None

    def fill_device(self, info: Any, dev_props: dict[str, Any], options: NetOptions):
        """Sets the addresses of `info` from device properties."""
        if NetOptions.HWADDR in options:
            info.hwaddr = dev_props.get("HwAddress")

        if NetOptions.IPV4 in options:
            info.ipv4 = self.get_ipaddr(dev_props.get("Ip4Config"), 4)

        if NetOptions.IPV6 in options:
            info.ipv6 = self.get_ipaddr(dev_props.get("Ip6Config"), 6)


class ConnManMixin(DBusFacade):
    BASE_SVC = "net.connman"
    BASE_OBJ = "/"
    BASE_IFACE = "net.connman.Manager"

    def get_service(self, iface: str, svc_type: str) -> dict[str, Any] | None:
        """Returns the properties of the connected service of the interface."""
        (services,) = self.call(
            self.BASE_OBJ, self.BASE_IFACE, "GetServices", None, "(a(oa{sv}))"
        )

        for _, svc in services:
            if svc.get("Type") != svc_type or svc.get("State") not in (
                "ready",
                "online",
            ):
                continue

            if (svc.get("Ethernet") or {}).get("Interface") == iface:
                return svc

        return None

    @staticmethod
    def fill_service(info: Any, svc: dict[str, Any], options: NetOptions):
        """Sets the addresses of `info` from service properties."""
        if NetOptions.HWADDR in options:
            info.hwaddr = svc["Ethernet"].get("Address")

        if NetOptions.IPV4 in options:
            if ipv4_obj := svc.get("IPv4"):
                info.ipv4 = ipv4_obj.get("Address")

        if NetOptions.IPV6 in options:
            if ipv6_obj := svc.get("IPv6"):
                info.ipv6 = ipv6_obj.get("Address")


class PolledNetWidget(WidgetBase):
    """Polls the state of the `iface` network interface, or of the first one
    detected, using the `backend`, or the one detected. The widget is hidden
    while there is no interface."""

    POLLING = True
    UNIQUE = False

    # backend name -> class, and the D-Bus services that select backends
    BACKEND_MAP: dict[str, type]
    BACKEND_SERVICES: dict[str, str]
    DEFAULT_BACKEND = "unmanaged"
    FMT_FIELDS: list[str]

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        backend = self.cfg.get("backend", "auto")

        if backend != "auto" and backend not in self.BACKEND_MAP:
            backends = ", ".join(repr(b) for b in ("auto", *self.BACKEND_MAP))
            raise BarConfigError(f"unknown backend '{backend}', not one of: {backends}")

        self.backend_name = backend
        self.backend: Any = None if backend == "auto" else self.BACKEND_MAP[backend]()

        iface = self.cfg.get("iface")

        if iface is not None and (not isinstance(iface, str) or not iface):
            raise BarConfigError("'iface' must be a network interface name")

        self.iface: str | None = iface

        self.qry_options = NetOptions.NONE

        for fld in set(self.formatter.get_fields(self.content.label)):
            if fld not in self.FMT_FIELDS:
                raise BarConfigError(f"unknown label field: {fld}")
            self.qry_options |= NetOptions.for_name(fld)

    def detect_iface(self) -> str | None:
        raise NotImplementedError()

    def empty_info(self, iface: str) -> Any:
        """The information shown while the interface or service is not
        available."""
        raise NotImplementedError()

    def info_ramp_level(self, info: Any) -> int:
        raise NotImplementedError()

    async def _detect_backend(self):
        name = await anyio.to_thread.run_sync(
            detect_backend, self.BACKEND_SERVICES, self.DEFAULT_BACKEND
        )
        logging.debug("widget '%s': using backend '%s'", self.widget_name, name)
        self.backend = self.BACKEND_MAP[name]()

    async def run(self):
        last_info = None

        while await self.sleep_interval():
            if self.backend is None:
                await self._detect_backend()

            # e.g. an USB adapter may be plugged in later
            if (iface := self.iface or self.detect_iface()) is None:
                self.set_available_i(False)
                continue

            try:
                info = await anyio.to_thread.run_sync(
                    self.backend.get_info, iface, self.qry_options
                )
            except DBusUnavailable:
                # e.g. the interface is removed, or the service is restarting
                info = self.empty_info(iface)

            self.set_available_i(True)

            if info != last_info:
                last_info = info
                self.set_new_content_i(self.info_ramp_level(info), **asdict(info))
