from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import spread_index

from ._net import NetOptions, PolledNetWidget, wireless_interfaces
from ._wifi import (
    ConnManBackend,
    IWDBackend,
    NetworkManagerBackend,
    UnmanagedBackend,
    WifiInfo,
    WPASupplicantBackend,
)


class WidgetWifi(PolledNetWidget):
    """Shows the state of a wireless network interface. The first ramp entry
    is shown when there is no signal, the rest is spread over the signal
    strength."""

    TYPE = "wifi"
    DEFAULT_TOOLTIP = "{iface}: {ssid} {percent}% {ipv4}"
    MAX_SIGNAL = 100

    BACKEND_MAP = {
        "NetworkManager": NetworkManagerBackend,
        "iwd": IWDBackend,
        "connman": ConnManBackend,
        "wpa_supplicant": WPASupplicantBackend,
        "unmanaged": UnmanagedBackend,
    }

    # NetworkManager and ConnMan may manage iwd or wpa_supplicant, they come
    # first
    BACKEND_SERVICES = {
        "org.freedesktop.NetworkManager": "NetworkManager",
        "net.connman": "connman",
        "net.connman.iwd": "iwd",
        "fi.w1.wpa_supplicant1": "wpa_supplicant",
    }

    FMT_FIELDS = [
        "ssid",
        "ipv4",
        "ipv6",
        "security",
        "hwaddr",
        "percent",
        "rssi",
        "ramp",
        "iface",
    ]

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        if self.ramp:
            self.qry_options |= NetOptions.SIGNAL

        if self.qry_options == NetOptions.NONE:
            raise BarConfigError("no known format fields for label")

    def detect_iface(self) -> str | None:
        return next(iter(wireless_interfaces()), None)

    def empty_info(self, iface: str) -> WifiInfo:
        return WifiInfo(iface, None, None, None)

    def info_ramp_level(self, info: WifiInfo) -> int:
        if info.percent is not None and info.percent >= 0:
            return info.percent
        return -1

    def ramp_index(self, ramp_level: int) -> int | None:
        if not self.ramp:
            return None

        if ramp_level < 0 or len(self.ramp) == 1:
            return 0

        return 1 + spread_index(ramp_level, self.MAX_SIGNAL, len(self.ramp) - 1)
