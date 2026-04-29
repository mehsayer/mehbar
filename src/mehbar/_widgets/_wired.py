import enum
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anyio
import psutil
from gi.repository import Gio

from mehbar._internals import DBusFacade
from mehbar.exceptions import BarConfigError, CapabilityError
from mehbar.widgets import Widget


@enum.verify(enum.NAMED_FLAGS)
class WiredOptions(enum.Flag):
    NONE = enum.auto()
    HWADDR = enum.auto()
    NAME = enum.auto()
    IPV4 = enum.auto()
    IPV6 = enum.auto()

    @classmethod
    def for_name(cls, name: str):

        ret = cls.NONE

        for member in cls:
            if member.name.casefold() == name.casefold():
                ret = member
                break
        return ret


@dataclass
class WiredInfo:
    iface: str | None
    name: str | None
    powered: bool
    connected: bool
    hwaddr: str | None = field(default=None)
    ipv4: str | None = field(default=None)
    ipv6: str | None = field(default=None)


def fill_missing_r(info: WiredInfo, options: WiredOptions):

    get_hwaddr = WiredOptions.HWADDR in options and info.hwaddr is None
    get_ipv4 = WiredOptions.IPV4 in options and info.ipv4 is None
    get_ipv6 = WiredOptions.IPV6 in options and info.ipv6 is None

    if get_hwaddr or get_ipv4 or get_ipv6:
        iface_info = psutil.net_if_addrs().get(info.iface, [])
        for snic in iface_info:
            if get_ipv4 and snic.family.name == "AF_INET":
                info.ipv4 = snic.address
            elif get_ipv6 and snic.family.name == "AF_INET6":
                if snic.address:
                    info.ipv6 = snic.address.rstrip("%" + info.iface)
            elif get_hwaddr and snic.family.name in ["AF_LINK", "AF_PACKET"]:
                if snic.address is not None:
                    info.hwaddr = snic.address.upper()

    return info


class WiredInfoQuery:
    async def get_info_(self, iface: str, options: WiredOptions):
        raise NotImplementedError()

    async def get_info(self, iface: str, options: WiredOptions):

        info = await self.get_info_(iface, options)

        return fill_missing_r(info, options)


class NetworkManagerBackend(DBusFacade, WiredInfoQuery):
    BASE_SVC = "org.freedesktop.NetworkManager"
    BASE_OBJ = "/org/freedesktop/NetworkManager"
    BASE_IFACE = "org.freedesktop.NetworkManager"

    ACT_CONN_IFACE = "org.freedesktop.NetworkManager.Connection.Active"
    AP_IFACE = "org.freedesktop.NetworkManager.AccessPoint"
    CONN_IFACE = "org.freedesktop.NetworkManager.Settings.Connection"
    DEV_IFACE = "org.freedesktop.NetworkManager.Device"
    DEV_WL_IFACE = "org.freedesktop.NetworkManager.Device.Wireless"
    SETTINGS_IFACE = "org.freedesktop.NetworkManager.Settings"
    SETTINGS_OBJ = "/org/freedesktop/NetworkManager/Settings"

    async def p_get_ipaddr_async(self, proxy: Gio.DBusProxy, ip_ver: int) -> str:

        ret = None

        addr_cfg_obj = self.p_get_prop(proxy, f"Ip{ip_ver}Config")
        addr_data = await self.get_prop_async(
            self.BASE_IFACE + f".IP{ip_ver}Config", addr_cfg_obj, "AddressData"
        )
        if addr_data is not None:
            ret = addr_data[0]["address"]
        return ret

    async def get_info_(self, iface: str, options: WiredOptions):

        hwaddr = None
        ipv4 = None
        ipv6 = None
        security = None
        ssid = None
        pcnt = None

        dev_prop = await self.get_prop_async(self.BASE_IFACE, self.BASE_OBJ, "Devices")

        if dev_prop is None or not dev_prop:
            raise RuntimeError("NetworkManager: no devices available")

        for dev_obj in dev_prop:
            dev_proxy = await self.p_new_async(self.DEV_IFACE, dev_obj)

            if self.p_get_prop(dev_proxy, "Interface") == iface:
                if WiredOptions.HWADDR in options:
                    hwaddr = self.p_get_prop(dev_proxy, "HwAddress")

                if WiredOptions.SIGNAL & options:
                    ap_obj = await self.get_prop_async(
                        self.DEV_WL_IFACE, dev_obj, "ActiveAccessPoint"
                    )
                    pcnt = await self.get_prop_async(self.AP_IFACE, ap_obj, "Strength")

                o_act_conn = self.p_get_prop(dev_proxy, "ActiveConnection")

                if o_act_conn is not None:
                    p_act_conn = await self.p_new_async(self.ACT_CONN_IFACE, o_act_conn)

                    conn_type = self.p_get_prop(p_act_conn, "Type")

                    if conn_type == "802-11-wireless":
                        if WiredOptions.IPV4 in options:
                            ipv4 = await self.p_get_ipaddr_async(p_act_conn, 4)

                        if WiredOptions.IPV6 in options:
                            ipv6 = await self.p_get_ipaddr_async(p_act_conn, 6)

                        o_conn = self.p_get_prop(p_act_conn, "Connection")

                        if (WiredOptions.SSID | WiredOptions.SECURITY) & options:
                            if o_conn is not None:
                                method = self.CONN_IFACE + ".GetSettings"

                                result = await self.new_call_async(
                                    self.CONN_IFACE, o_conn, method
                                )

                                if (wlan := result.get("802-11-wireless")) is not None:
                                    if WiredOptions.SSID in options:
                                        _ssid = "".join(
                                            chr(c) for c in wlan.get("ssid", [])
                                        )
                                        if _ssid:
                                            ssid = _ssid

                                    if WiredOptions.SECURITY in options:
                                        sec_type = wlan.get("security")
                                        if sec_type is not None:
                                            if (
                                                sec := result.get(sec_type)
                                            ) is not None:
                                                if _sec := sec.get("key-mgmt"):
                                                    security = _sec.upper()
                break

        return WiredInfo(iface, ssid, None, pcnt, hwaddr, security, ipv4, ipv6)


class UnmanagedBackend(WiredInfoQuery):
    def __init__(self, *_):
        pass

    async def get_info_(self, iface: str, options: WiredOptions):

        base_path = Path(f"/sys/class/net/{iface}")

        pwrd = False
        connd = False

        async with await anyio.open_file(base_path / "operstate", "r") as fhandle:
            if (await fhandle.readline()).strip() == "up":
                pwrd = True

        async with await anyio.open_file(base_path / "carrier", "r") as fhandle:
            if (await fhandle.readline()).strip() == "1":
                connd = True

        return WiredInfo(iface, None, pwrd, connd)


class ConnManBackend(DBusFacade, WiredInfoQuery):
    BASE_SVC = "net.connman"
    BASE_OBJ = "/"
    BASE_IFACE = "net.connman.Manager"

    TECH_OBJ = "/net/connman/technology/ethernet"
    TECH_IFACE = "net.connman.Technology"

    async def get_info_(self, iface: str, options: WiredOptions):

        hwaddr = None
        ipv4 = None
        name = None
        ipv6 = None
        ready = False

        tech_props = await self.new_call_async(
            self.TECH_IFACE, self.TECH_OBJ, "GetProperties"
        )

        connd = tech_props.get("Connected", False)
        pwrd = tech_props.get("Powered", False)

        services = await self.new_call_async(
            self.BASE_IFACE, self.BASE_OBJ, "GetServices"
        )

        if pwrd:
            if services is None or not services:
                raise CapabilityError("no 'connman' services found")

            for _, svc in services:
                if svc.get("Type") == "ethernet" and svc.get("State") == "ready":
                    ready &= True
                    if eth_obj := svc.get("Ethernet"):
                        if eth_obj.get("Interface") == iface:
                            if WiredOptions.NAME in options:
                                name = svc.get("Name")

                            if WiredOptions.HWADDR in options:
                                hwaddr = eth_obj.get("Address")

                            if WiredOptions.IPV4 in options:
                                if ipv4_obj := svc.get("IPv4"):
                                    ipv4 = ipv4_obj.get("Address")

                            if WiredOptions.IPV6 in options:
                                if ipv6_obj := svc.get("IPv6"):
                                    ipv6 = ipv6_obj.get("Address")

                            break

        return WiredInfo(iface, name, pwrd, connd, hwaddr, ipv4, ipv6)


class WidgetWired(Widget):
    BACKEND_MAP = {
        "NetworkManager": NetworkManagerBackend,
        "connman": ConnManBackend,
        "unmanaged": UnmanagedBackend,
    }

    FMT_FIELDS = ["ipv4", "ipv6", "hwaddr", "ramp"]

    def __init__(
        self,
        interval: int,
        iface: str,
        backend: str,
        label_format: str,
        ramp: list[str] | None = None,
    ):
        super().__init__(interval, label_format, ramp)

        if backend not in self.BACKEND_MAP:
            raise BarConfigError(f"unknown backend: {backend}")

        self.dbus_iface = self.BACKEND_MAP[backend](None)

        self.iface = iface
        self.ramps = []
        if ramp is not None:
            self.ramps = ramp[:3]

        self.ramps.extend([None] * (3 - len(self.ramps)))

        self.qry_options = WiredOptions.NONE

        fld_cnt = 0

        for fld in set(self.formatter.get_fields(label_format)):
            if fld in self.FMT_FIELDS:
                fld_cnt += 1
                opt = WiredOptions.for_name(fld)

                if opt is not WiredOptions.NONE:
                    self.qry_options |= opt
            else:
                raise BarConfigError(f"unknown label field: {fld}")

        if fld_cnt == 0:
            raise BarConfigError("no known format fields for label")

    async def run(self):

        while await self.sleep_interval():
            info = await self.dbus_iface.get_info(self.iface, self.qry_options)
            ramp = None

            if self.ramps:
                if info.connected and info.powered:
                    ramp = self.ramps[2]
                elif not info.connected:
                    ramp = self.ramps[1]
                else:
                    ramp = self.ramps[0]

            self.format_label_idle(ramp=ramp, **asdict(info))
