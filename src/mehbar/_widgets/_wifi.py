from dataclasses import dataclass, field

from gi.repository import GLib  # type: ignore

from mehbar.exceptions import CapabilityError

from ._dbus_facade import DBusFacade, DBusUnavailable, is_null_path
from ._net import ConnManMixin, NetOptions, NetworkManagerMixin, fill_addresses

# signal strength scale of NetworkManager, so that percentages and levels of
# all backends match
MIN_RSSI = -100
MAX_RSSI = -40


@dataclass
class WifiInfo:
    iface: str | None
    ssid: str | None
    rssi: int | None
    percent: int | None
    hwaddr: str | None = field(default=None)
    security: str | None = field(default=None)
    ipv4: str | None = field(default=None)
    ipv6: str | None = field(default=None)


def rssi_to_strength(rssi: float | int) -> int:
    """Maps the signal level in dBm to percent, as NetworkManager does."""
    rssi = max(MIN_RSSI, min(rssi, MAX_RSSI))
    return 100 - int(100 * (MAX_RSSI - rssi) / (MAX_RSSI - MIN_RSSI))


def strength_to_rssi(percent: float | int) -> int:
    return round(MAX_RSSI - (100 - percent) * (MAX_RSSI - MIN_RSSI) / 100)


def decode_ssid(ssid: bytes | list[int] | None) -> str | None:
    if not ssid:
        return None
    return bytes(ssid).decode("utf-8", "replace")


class WifiInfoQuery:
    """Backends are synchronous, call `get_info()` in a worker thread."""

    def get_info_(self, iface: str, options: NetOptions) -> WifiInfo:
        raise NotImplementedError()

    def get_info(self, iface: str, options: NetOptions) -> WifiInfo:
        info = self.get_info_(iface, options)

        if NetOptions.SIGNAL & options:
            if info.rssi is None and info.percent is not None:
                info.rssi = strength_to_rssi(info.percent)
            elif info.percent is None and info.rssi is not None:
                info.percent = rssi_to_strength(info.rssi)

        return fill_addresses(info, options)


class NetworkManagerBackend(NetworkManagerMixin, WifiInfoQuery):
    AP_IFACE = "org.freedesktop.NetworkManager.AccessPoint"
    CONN_IFACE = "org.freedesktop.NetworkManager.Settings.Connection"
    DEV_WL_IFACE = "org.freedesktop.NetworkManager.Device.Wireless"

    def get_info_(self, iface: str, options: NetOptions) -> WifiInfo:
        info = WifiInfo(iface, None, None, None)

        dev_obj, dev_props = self.get_device(iface)
        self.fill_device(info, dev_props, options)

        if (NetOptions.SIGNAL | NetOptions.SSID) & options:
            ap_obj = self.get_prop(dev_obj, self.DEV_WL_IFACE, "ActiveAccessPoint")

            if not is_null_path(ap_obj):
                ap_props = self.get_all_props(ap_obj, self.AP_IFACE)
                info.percent = ap_props.get("Strength")
                info.ssid = decode_ssid(ap_props.get("Ssid"))

        if NetOptions.SECURITY in options:
            act_conn_obj = dev_props.get("ActiveConnection")

            if not is_null_path(act_conn_obj):
                conn_obj = self.get_prop(
                    act_conn_obj, self.ACT_CONN_IFACE, "Connection"
                )
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

    UNAVAILABLE_ERROR_NAMES = frozenset({"fi.w1.wpa_supplicant1.InterfaceUnknown"})

    def get_info_(self, iface: str, options: NetOptions) -> WifiInfo:
        info = WifiInfo(iface, None, None, None)

        (iface_obj,) = self.call(
            self.BASE_OBJ,
            self.BASE_IFACE,
            "GetInterface",
            GLib.Variant("(s)", (iface,)),
            "(o)",
        )

        iface_props = self.get_all_props(iface_obj, self.IFACE_IFACE)

        if NetOptions.HWADDR in options:
            if hwaddr := iface_props.get("MACAddress"):
                info.hwaddr = ":".join(f"{n:02X}" for n in hwaddr)

        bss_obj = iface_props.get("CurrentBSS")

        if (NetOptions.SSID | NetOptions.SIGNAL) & options:
            if not is_null_path(bss_obj):
                bss_props = self.get_all_props(bss_obj, self.IFACE_BSS)
                info.ssid = decode_ssid(bss_props.get("SSID"))
                info.rssi = bss_props.get("Signal")

        if NetOptions.SECURITY in options:
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

    def get_info_(self, iface: str, options: NetOptions) -> WifiInfo:
        rssi = None

        if NetOptions.SIGNAL & options:
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

            # the level is 0 while disconnected, it is negative in dBm otherwise
            if rssi is not None and rssi >= 0:
                rssi = None

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
            except DBusUnavailable:
                raise
            except GLib.Error:
                # e.g. iwd is built without the diagnostics interface
                self._has_diagnostics = False

        (networks,) = self.call(
            stn_path, self.STATION_IFACE, "GetOrderedNetworks", None, "(a(on))"
        )

        for netw_path, netw_rssi in networks:
            if netw_path == conn_netw:
                return round(netw_rssi / 100)

        return None

    def get_info_(self, iface: str, options: NetOptions) -> WifiInfo:
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

            if NetOptions.HWADDR in options:
                if hwaddr := dev_props.get("Address"):
                    info.hwaddr = hwaddr.upper()

            stn_props = interfaces.get(self.STATION_IFACE) or {}

            if conn_netw := stn_props.get("ConnectedNetwork"):
                if NetOptions.SIGNAL & options:
                    info.rssi = self._get_rssi(stn_path, conn_netw)

                netw_props = objects.get(conn_netw, {}).get(self.NETWORK_IFACE, {})

                if NetOptions.SSID in options:
                    info.ssid = netw_props.get("Name")

                if NetOptions.SECURITY in options:
                    if security := netw_props.get("Type"):
                        info.security = security.upper()
            break

        return info


class ConnManBackend(ConnManMixin, WifiInfoQuery):
    def get_info_(self, iface: str, options: NetOptions) -> WifiInfo:
        info = WifiInfo(iface, None, None, None)

        if (svc := self.get_service(iface, "wifi")) is None:
            return info

        self.fill_service(info, svc, options)

        if NetOptions.SECURITY in options:
            if sec := svc.get("Security"):
                info.security = ", ".join(sec).upper()

        if NetOptions.SSID in options:
            info.ssid = svc.get("Name")

        if NetOptions.SIGNAL & options:
            if (strength := svc.get("Strength")) is not None:
                info.percent = round(strength)

        return info
