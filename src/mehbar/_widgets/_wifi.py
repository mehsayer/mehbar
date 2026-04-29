from dataclasses import asdict

from mehbar._internals import (
    ConnManBackend,
    IWDBackend,
    NetworkManagerBackend,
    UnmanagedBackend,
    WifiOptions,
    WPASupplicantBackend,
)
from mehbar.exceptions import BarConfigError
from mehbar.widgets import Widget


class WidgetWifiSignal(Widget):
    MAX_SIGNAL = 100

    BACKEND_MAP = {
        "NetworkManager": NetworkManagerBackend,
        "iwd": IWDBackend,
        "connman": ConnManBackend,
        "wpa_supplicant": WPASupplicantBackend,
        "unmanaged": UnmanagedBackend,
    }

    FMT_FIELDS = [
        "ssid",
        "ipv4",
        "ipv6",
        "security",
        "hwaddr",
        "percentage",
        "rssi",
        "ramp",
    ]

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

        self.qry_options = WifiOptions.NONE

        for fld in set(self.formatter.get_fields(label_format)):
            if fld in self.FMT_FIELDS:
                opt = WifiOptions.for_name(fld)

                if opt is not WifiOptions.NONE:
                    self.qry_options |= opt
            else:
                raise BarConfigError(f"unknown label field: {fld}")

        if self.qry_options == WifiOptions.NONE:
            raise BarConfigError("no known format fields for label")

        if ramp is not None and (nramp := len(ramp) - 2) > 0:
            self.ramps.extend(ramp[:2])

            for sig in range(2, self.MAX_SIGNAL + 2):
                ramp_idx = int(
                    min(sig, self.MAX_SIGNAL - 1) / (self.MAX_SIGNAL / nramp)
                )
                ramp_val = ramp[2:][ramp_idx]

                self.ramps.append(ramp_val)

    async def run(self):

        info = None

        while await self.sleep_interval():
            info = await self.dbus_iface.get_info(self.iface, self.qry_options)

            if not info.matches(self._last_value):
                self._last_value = info
                ramp = None

                if self.ramps and info.percentage is not None:
                    ramp = self.ramps[info.percentage]

                self.format_label_idle(ramp=ramp, **asdict(info))
