#!/usr/bin/env python3

from mehbar._internals import DBusFacade
import enum
from dataclasses import dataclass, field


MIN_RSSI = -100
MAX_RSSI = -30


@enum.verify(enum.NAMED_FLAGS)
class WifiOptions(enum.Flag):
    NONE = enum.auto()
    HWADDR = enum.auto()
    IPV4 = enum.auto()
    IPV6 = enum.auto()
    PERCENTAGE = enum.auto()
    RSSI = enum.auto()
    SECURITY = enum.auto()
    SSID = enum.auto()
    SIGNAL = PERCENTAGE | RSSI

    @classmethod
    def for_name(cls, name: str):
        ret = cls.NONE

        for member in cls:
            if member.name.casefold() == name.casefold():
                ret = member
                break
        return ret


@dataclass
class WifiInfo:
    iface: str | None
    ssid: str | None
    rssi: int | None
    percentage: int | None
    hwaddr: str | None = field(default=None)
    security: str | None = field(default=None)
    ipv4: str | None = field(default=None)
    ipv6: str | None = field(default=None)


    def matches(self, other: WifiInfo | int | None):
        ret = False
        if other is not None:
            if isinstance(other, WifiInfo):

                # TODO: add None checks
                ret = self.iface == other.iface and self.ssid == other.ssid \
                      and (self.rssi == other.rssi or self.percentage == other.percentage) \
                      and (self.ipv4 == other.ipv4 or self.ipv6 == other.ipv6)
            else:
                raise TypeError("expected 'WifiInfo' or 'int'")

        return ret


def rssi_to_strength(rssi: float | int) -> int:
    return round(100 * (1 - (MAX_RSSI - rssi) / (MAX_RSSI - MIN_RSSI)))


def strength_to_rssi(percentage: float | int) -> int:
    return round((percentage * (MAX_RSSI - MIN_RSSI)) / 100 + MIN_RSSI)


def fill_missing_r(info: WifiInfo, options: WifiOptions):

    get_signal = WifiOptions.SIGNAL & options and (
        info.rssi is None or info.percentage is None
    )
    get_hwaddr = WifiOptions.HWADDR in options and info.hwaddr is None
    get_ipv4 = WifiOptions.IPV4 in options and info.ipv4 is None
    get_ipv6 = WifiOptions.IPV6 in options and info.ipv6 is None

    if get_signal:
        if info.rssi is None and info.percentage is not None:
            info.rssi = strength_to_rssi(info.percentage)
        elif info.percentage is None and info.rssi is not None:
            info.percentage = rssi_to_strength(info.rssi)

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


class WifiInfoQuery:

    async def get_info_(self, iface: str, options: WifiOptions):
        raise NotImplementedError()

    async def get_info(self, iface: str, options: WifiOptions):

        info = await self.get_info_(iface, options)

        return fill_missing_r(info, options)


class NetworkManagerBackend(DBusFacade, WifiInfoQuery):

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

    async def get_info_(self, iface: str, options: WifiOptions):

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

                if WifiOptions.HWADDR in options:
                    hwaddr = self.p_get_prop(dev_proxy, "HwAddress")

                if WifiOptions.SIGNAL & options:
                    ap_obj = await self.get_prop_async(
                        self.DEV_WL_IFACE, dev_obj, "ActiveAccessPoint"
                    )
                    pcnt = await self.get_prop_async(self.AP_IFACE, ap_obj, "Strength")

                o_act_conn = self.p_get_prop(dev_proxy, "ActiveConnection")

                if o_act_conn is not None:

                    p_act_conn = await self.p_new_async(self.ACT_CONN_IFACE, o_act_conn)

                    conn_type = self.p_get_prop(p_act_conn, "Type")

                    if conn_type == "802-11-wireless":

                        if WifiOptions.IPV4 in options:
                            ipv4 = await self.p_get_ipaddr_async(p_act_conn, 4)

                        if WifiOptions.IPV6 in options:
                            ipv6 = await self.p_get_ipaddr_async(p_act_conn, 6)

                        o_conn = self.p_get_prop(p_act_conn, "Connection")

                        if (WifiOptions.SSID | WifiOptions.SECURITY) & options:

                            if o_conn is not None:

                                method = self.CONN_IFACE + ".GetSettings"

                                result = await self.new_call_async(
                                    self.CONN_IFACE, o_conn, method
                                )

                                if (wlan := result.get("802-11-wireless")) is not None:
                                    if WifiOptions.SSID in options:
                                        _ssid = "".join(
                                            chr(c) for c in wlan.get("ssid", [])
                                        )
                                        if _ssid:
                                            ssid = _ssid

                                    if WifiOptions.SECURITY in options:
                                        sec_type = wlan.get("security")
                                        if sec_type is not None:

                                            if (
                                                sec := result.get(sec_type)
                                            ) is not None:
                                                if _sec := sec.get("key-mgmt"):
                                                    security = _sec.upper()
                break

        return WifiInfo(iface, ssid, None, pcnt, hwaddr, security, ipv4, ipv6)


class WPASupplicantBackend(DBusFacade, WifiInfoQuery):
    BASE_SVC = "fi.w1.wpa_supplicant1"
    BASE_OBJ = "/fi/w1/wpa_supplicant1"
    BASE_IFACE = "fi.w1.wpa_supplicant1"

    IFACE_IFACE = "fi.w1.wpa_supplicant1.Interface"
    IFACE_NETWORK = "fi.w1.wpa_supplicant1.Network"
    IFACE_BSS = "fi.w1.wpa_supplicant1.BSS"

    async def get_info_(self, iface: str, options: WifiOptions):

        o_iface = await self.new_call_async(
            self.BASE_IFACE, self.BASE_OBJ, self.BASE_IFACE + ".GetInterface", iface
        )

        p_iface = await self.p_new_async(self.IFACE_IFACE, o_iface)

        o_netw = self.p_get_prop(p_iface, "CurrentNetwork")

        rssi = None
        hwaddr = None
        security = None
        ssid = None

        if WifiOptions.HWADDR in options:
            _hwaddr = self.p_get_prop(p_iface, "MACAddress")

            if _hwaddr is not None and _hwaddr:
                hwaddr = ":".join(f"{n:02X}" for n in _hwaddr)

        o_bss = self.p_get_prop(p_iface, "CurrentBSS")

        p_bss = await self.p_new_async(self.IFACE_BSS, o_bss)

        if WifiOptions.SSID in options:
            if _ssid := self.p_get_prop(p_bss, "SSID"):
                ssid = "".join(chr(c) for c in _ssid)

        if WifiOptions.SIGNAL & options:
            rssi = self.p_get_prop(p_bss, "Signal")

        if WifiOptions.SECURITY in options:

            p_netw = await self.p_new_async(self.IFACE_NETWORK, o_netw)

            if self.p_get_prop(p_netw, "Enabled"):
                netw_props = self.p_get_prop(p_netw, "Properties")

                if _security := netw_props.get("key_mgmt"):
                    security = ", ".join(_security.split())

        return WifiInfo(iface, ssid, rssi, None, hwaddr, security)


class UnmanagedBackend(WifiInfoQuery):

    def __init__(self, *_):
        pass

    async def get_info_(self, iface: str, options: WifiOptions):

        rssi = None

        if WifiOptions.SIGNAL & options:
            async with await anyio.open_file("/proc/net/wireless", "r") as fhandle:
                async for ln in fhandle:
                    ln_split = ln.split(maxsplit=4)
                    if len(ln_split) > 4:
                        _iface, _, _, _rssi, _ = ln_split

                        if _iface.startswith(iface):
                            rssi = int(_rssi[:-1])
                            break

        return WifiInfo(iface, None, rssi, None, None)


class IWDBackend(DBusFacade, WifiInfoQuery):

    BASE_IFACE = "org.freedesktop.DBus.ObjectManager"
    BASE_SVC = "net.connman.iwd"
    BASE_OBJ = "/"
    DEVICE_IFACE = "net.connman.iwd.Device"
    STATION_IFACE = "net.connman.iwd.Station"
    NETWORK_IFACE = "net.connman.iwd.Network"

    async def get_info_(self, iface: str, options: WifiOptions):

        hwaddr = None
        rssi = None
        ssid = None
        security = None

        objects = await self.new_call_async(
            self.BASE_IFACE, self.BASE_OBJ, "GetManagedObjects"
        )

        if objects is None or not objects:
            raise CapabilityError("no 'iwd' managed objects found")

        opts_props = WifiOptions.SIGNAL | WifiOptions.SSID | WifiOptions.SECURITY

        for stn_path, interfaces in objects.items():
            if dev_props := interfaces.get(self.DEVICE_IFACE):
                if dev_props.get("Name") == iface:
                    if not dev_props.get("Powered", False):
                        continue

                    if WifiOptions.HWADDR in options:
                        hwaddr = dev_props.get("Address", "unknown")

                    need_props = opts_props & options

                    if need_props and (stn_props := interfaces.get(self.STATION_IFACE)):
                        if conn_netw := stn_props.get("ConnectedNetwork"):

                            method = self.STATION_IFACE + ".GetOrderedNetworks"

                            networks = await self.new_call_async(
                                self.NETWORK_IFACE, stn_path, method
                            )

                            if WifiOptions.SIGNAL & options and networks is not None:
                                for netw_path, netw_rssi in networks:
                                    if netw_path == conn_netw:
                                        rssi = round(netw_rssi / 100)
                                        break

                            if (WifiOptions.SSID | WifiOptions.SECURITY) & options:

                                p_ap = await self.p_new_async(
                                    self.NETWORK_IFACE, conn_netw
                                )

                                if WifiOptions.SSID in options:
                                    _ssid = p_ap.get_cached_property("Name")
                                    ssid = _ssid.unpack() if _ssid else None

                                if WifiOptions.SECURITY in options:
                                    _security = p_ap.get_cached_property("Type")
                                    security = _security.unpack() if _security else None
                            break

            if hwaddr is not None:
                hwaddr = hwaddr.upper()

            if security is not None:
                security = security.upper()

        return WifiInfo(iface, ssid, rssi, None, hwaddr, security)


class ConnManBackend(DBusFacade, WifiInfoQuery):

    BASE_SVC = "net.connman"
    BASE_OBJ = "/"
    BASE_IFACE = "net.connman.Manager"

    async def get_info_(self, iface: str, options: WifiOptions):

        hwaddr = None
        ipv4 = None
        ipv6 = None
        security = None
        ssid = None
        pcnt = None

        services = await self.new_call_async(
            self.BASE_IFACE, self.BASE_OBJ, "GetServices"
        )

        if services is None or not services:
            raise CapabilityError("no 'connman' services found")

        for _, svc in services:
            if svc.get("Type") == "wifi" and svc.get("State") == "ready":
                if eth_obj := svc.get("Ethernet"):
                    if eth_obj.get("Interface") == iface:
                        if WifiOptions.SECURITY in options:
                            if sec := svc.get("Security"):
                                security = ", ".join(sec).upper()

                        if WifiOptions.SSID in options:
                            ssid = svc.get("Name")

                        if WifiOptions.SIGNAL in options:
                            pcnt = round(svc.get("Strength", 0))

                        if WifiOptions.HWADDR in options:
                            hwaddr = eth_obj.get("Address")

                        if WifiOptions.IPV4 in options:
                            if ipv4_obj := svc.get("IPv4"):
                                ipv4 = ipv4_obj.get("Address")

                        if WifiOptions.IPV6 in options:
                            if ipv6_obj := svc.get("IPv6"):
                                ipv6 = ipv6_obj.get("Address")

                        break

        return WifiInfo(iface, ssid, None, pcnt, hwaddr, security, ipv4, ipv6)
