#!/usr/bin/env python3

import enum
from dataclasses import dataclass, field

from gi.repository import GLib  # type: ignore

from mehbar.exceptions import CapabilityError

from ._dbus_facade import DBusFacade, is_null_path

MIN_RSSI = -100
MAX_RSSI = -30


class WifiOptions(enum.Flag):
    NONE = 0
    HWADDR = enum.auto()
    IPV4 = enum.auto()
    IPV6 = enum.auto()
    PERCENTAGE = enum.auto()
    RSSI = enum.auto()
    SECURITY = enum.auto()
    SSID = enum.auto()
    SIGNAL = PERCENTAGE | RSSI

    @classmethod
    def for_name(cls, name: str) -> "WifiOptions":
        return cls.__members__.get(name.upper(), cls.NONE)


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


def rssi_to_strength(rssi: float | int) -> int:
    pcnt = round(100 * (1 - (MAX_RSSI - rssi) / (MAX_RSSI - MIN_RSSI)))
    return max(0, min(pcnt, 100))


def strength_to_rssi(percentage: float | int) -> int:
    return round((percentage * (MAX_RSSI - MIN_RSSI)) / 100 + MIN_RSSI)


def decode_ssid(ssid: bytes | list[int] | None) -> str | None:
    if not ssid:
        return None
    return bytes(ssid).decode("utf-8", "replace")


def fill_missing(info: WifiInfo, options: WifiOptions) -> WifiInfo:
    from psutil import net_if_addrs

    if WifiOptions.SIGNAL & options:
        if info.rssi is None and info.percentage is not None:
            info.rssi = strength_to_rssi(info.percentage)
        elif info.percentage is None and info.rssi is not None:
            info.percentage = rssi_to_strength(info.rssi)

    get_hwaddr = WifiOptions.HWADDR in options and info.hwaddr is None
    get_ipv4 = WifiOptions.IPV4 in options and info.ipv4 is None
    get_ipv6 = WifiOptions.IPV6 in options and info.ipv6 is None

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


class WifiInfoQuery:
    """Backends are synchronous, call `get_info()` in a worker thread."""

    def get_info_(self, iface: str, options: WifiOptions) -> WifiInfo:
        raise NotImplementedError()

    def get_info(self, iface: str, options: WifiOptions) -> WifiInfo:
        return fill_missing(self.get_info_(iface, options), options)


class NetworkManagerBackend(DBusFacade, WifiInfoQuery):
    BASE_SVC = "org.freedesktop.NetworkManager"
    BASE_OBJ = "/org/freedesktop/NetworkManager"
    BASE_IFACE = "org.freedesktop.NetworkManager"

    AP_IFACE = "org.freedesktop.NetworkManager.AccessPoint"
    ACT_CONN_IFACE = "org.freedesktop.NetworkManager.Connection.Active"
    CONN_IFACE = "org.freedesktop.NetworkManager.Settings.Connection"
    DEV_IFACE = "org.freedesktop.NetworkManager.Device"
    DEV_WL_IFACE = "org.freedesktop.NetworkManager.Device.Wireless"

    def get_ipaddr(self, cfg_obj: str | None, ip_ver: int) -> str | None:
        if is_null_path(cfg_obj):
            return None

        addr_data = self.get_prop(
            cfg_obj, f"{self.BASE_IFACE}.IP{ip_ver}Config", "AddressData"
        )
        return addr_data[0]["address"] if addr_data else None

    def get_info_(self, iface: str, options: WifiOptions) -> WifiInfo:
        info = WifiInfo(iface, None, None, None)

        (dev_obj,) = self.call(
            self.BASE_OBJ,
            self.BASE_IFACE,
            "GetDeviceByIpIface",
            GLib.Variant("(s)", (iface,)),
            "(o)",
        )

        dev_props = self.get_all_props(dev_obj, self.DEV_IFACE)

        if WifiOptions.HWADDR in options:
            info.hwaddr = dev_props.get("HwAddress")

        if WifiOptions.IPV4 in options:
            info.ipv4 = self.get_ipaddr(dev_props.get("Ip4Config"), 4)

        if WifiOptions.IPV6 in options:
            info.ipv6 = self.get_ipaddr(dev_props.get("Ip6Config"), 6)

        if (WifiOptions.SIGNAL | WifiOptions.SSID) & options:
            ap_obj = self.get_prop(dev_obj, self.DEV_WL_IFACE, "ActiveAccessPoint")

            if not is_null_path(ap_obj):
                ap_props = self.get_all_props(ap_obj, self.AP_IFACE)
                info.percentage = ap_props.get("Strength")
                info.ssid = decode_ssid(ap_props.get("Ssid"))

        if WifiOptions.SECURITY in options:
            act_conn_obj = dev_props.get("ActiveConnection")

            if not is_null_path(act_conn_obj):
                conn_obj = self.get_prop(act_conn_obj, self.ACT_CONN_IFACE, "Connection")
                (settings,) = self.call(
                    conn_obj, self.CONN_IFACE, "GetSettings", None, "(a{sa{sv}})"
                )

                if (wlan := settings.get("802-11-wireless")) is not None:
                    if (sec_type := wlan.get("security")) is not None:
                        if (sec := settings.get(sec_type)) is not None:
                            if key_mgmt := sec.get("key-mgmt"):
                                info.security = key_mgmt.upper()

        return info


class WPASupplicantBackend(DBusFacade, WifiInfoQuery):
    BASE_SVC = "fi.w1.wpa_supplicant1"
    BASE_OBJ = "/fi/w1/wpa_supplicant1"
    BASE_IFACE = "fi.w1.wpa_supplicant1"

    IFACE_IFACE = "fi.w1.wpa_supplicant1.Interface"
    IFACE_NETWORK = "fi.w1.wpa_supplicant1.Network"
    IFACE_BSS = "fi.w1.wpa_supplicant1.BSS"

    def get_info_(self, iface: str, options: WifiOptions) -> WifiInfo:
        info = WifiInfo(iface, None, None, None)

        (iface_obj,) = self.call(
            self.BASE_OBJ,
            self.BASE_IFACE,
            "GetInterface",
            GLib.Variant("(s)", (iface,)),
            "(o)",
        )

        iface_props = self.get_all_props(iface_obj, self.IFACE_IFACE)

        if WifiOptions.HWADDR in options:
            if hwaddr := iface_props.get("MACAddress"):
                info.hwaddr = ":".join(f"{n:02X}" for n in hwaddr)

        bss_obj = iface_props.get("CurrentBSS")

        if (WifiOptions.SSID | WifiOptions.SIGNAL) & options:
            if not is_null_path(bss_obj):
                bss_props = self.get_all_props(bss_obj, self.IFACE_BSS)
                info.ssid = decode_ssid(bss_props.get("SSID"))
                info.rssi = bss_props.get("Signal")

        if WifiOptions.SECURITY in options:
            netw_obj = iface_props.get("CurrentNetwork")

            if not is_null_path(netw_obj):
                netw_props = self.get_all_props(netw_obj, self.IFACE_NETWORK)

                if netw_props.get("Enabled"):
                    props = netw_props.get("Properties") or {}

                    if key_mgmt := props.get("key_mgmt"):
                        info.security = ", ".join(key_mgmt.split())

        return info


class UnmanagedBackend(WifiInfoQuery):
    PATH_WIRELESS = "/proc/net/wireless"

    def get_info_(self, iface: str, options: WifiOptions) -> WifiInfo:
        rssi = None

        if WifiOptions.SIGNAL & options:
            with open(self.PATH_WIRELESS, "r") as fhandle:
                for ln in fhandle:
                    ln_split = ln.split(maxsplit=4)
                    if len(ln_split) > 4:
                        _iface, _, _, _rssi, _ = ln_split

                        if _iface.rstrip(":") == iface:
                            try:
                                rssi = int(float(_rssi.rstrip(".")))
                            except ValueError:
                                pass
                            break

        return WifiInfo(iface, None, rssi, None)


class IWDBackend(DBusFacade, WifiInfoQuery):
    BASE_SVC = "net.connman.iwd"
    DEVICE_IFACE = "net.connman.iwd.Device"
    STATION_IFACE = "net.connman.iwd.Station"
    STATION_DIAG_IFACE = "net.connman.iwd.StationDiagnostic"
    NETWORK_IFACE = "net.connman.iwd.Network"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._has_diagnostics = True

    def _get_rssi(self, stn_path: str, conn_netw: str) -> int | None:
        # diagnostics provide the current signal level of the connected BSS
        if self._has_diagnostics:
            try:
                (diag,) = self.call(
                    stn_path, self.STATION_DIAG_IFACE, "GetDiagnostics", None, "(a{sv})"
                )
                if (rssi := diag.get("RSSI")) is not None:
                    return rssi
            except GLib.Error:
                self._has_diagnostics = False

        (networks,) = self.call(
            stn_path, self.STATION_IFACE, "GetOrderedNetworks", None, "(a(on))"
        )

        for netw_path, netw_rssi in networks:
            if netw_path == conn_netw:
                return round(netw_rssi / 100)

        return None

    def get_info_(self, iface: str, options: WifiOptions) -> WifiInfo:
        info = WifiInfo(iface, None, None, None)

        objects = self.get_managed_objects()

        if not objects:
            raise CapabilityError("no 'iwd' managed objects found")

        for stn_path, interfaces in objects.items():
            dev_props = interfaces.get(self.DEVICE_IFACE)

            if not dev_props or dev_props.get("Name") != iface:
                continue

            if not dev_props.get("Powered", False):
                break

            if WifiOptions.HWADDR in options:
                if hwaddr := dev_props.get("Address"):
                    info.hwaddr = hwaddr.upper()

            stn_props = interfaces.get(self.STATION_IFACE) or {}

            if conn_netw := stn_props.get("ConnectedNetwork"):
                if WifiOptions.SIGNAL & options:
                    info.rssi = self._get_rssi(stn_path, conn_netw)

                netw_props = objects.get(conn_netw, {}).get(self.NETWORK_IFACE, {})

                if WifiOptions.SSID in options:
                    info.ssid = netw_props.get("Name")

                if WifiOptions.SECURITY in options:
                    if security := netw_props.get("Type"):
                        info.security = security.upper()
            break

        return info


class ConnManBackend(DBusFacade, WifiInfoQuery):
    BASE_SVC = "net.connman"
    BASE_OBJ = "/"
    BASE_IFACE = "net.connman.Manager"

    def get_info_(self, iface: str, options: WifiOptions) -> WifiInfo:
        info = WifiInfo(iface, None, None, None)

        (services,) = self.call(
            self.BASE_OBJ, self.BASE_IFACE, "GetServices", None, "(a(oa{sv}))"
        )

        for _, svc in services:
            if svc.get("Type") != "wifi" or svc.get("State") not in ("ready", "online"):
                continue

            if not (eth_obj := svc.get("Ethernet")) or eth_obj.get("Interface") != iface:
                continue

            if WifiOptions.SECURITY in options:
                if sec := svc.get("Security"):
                    info.security = ", ".join(sec).upper()

            if WifiOptions.SSID in options:
                info.ssid = svc.get("Name")

            if WifiOptions.SIGNAL & options:
                info.percentage = round(svc.get("Strength", 0))

            if WifiOptions.HWADDR in options:
                info.hwaddr = eth_obj.get("Address")

            if WifiOptions.IPV4 in options:
                if ipv4_obj := svc.get("IPv4"):
                    info.ipv4 = ipv4_obj.get("Address")

            if WifiOptions.IPV6 in options:
                if ipv6_obj := svc.get("IPv6"):
                    info.ipv6 = ipv6_obj.get("Address")
            break

        return info
