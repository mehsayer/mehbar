from dataclasses import asdict

import anyio

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase

from ._wifi import (
    ConnManBackend,
    IWDBackend,
    NetworkManagerBackend,
    UnmanagedBackend,
    WifiOptions,
    WPASupplicantBackend,
)


class WidgetWifi(WidgetBase):
    MAX_SIGNAL = 100

    TYPE = "wifi"

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
        "iface",
    ]

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        backend = self.cfg.get("backend")

        if backend not in self.BACKEND_MAP:
            backends = ", ".join(repr(b) for b in self.BACKEND_MAP)
            raise BarConfigError(f"unknown backend '{backend}', not one of: {backends}")

        self.backend = self.BACKEND_MAP[backend]()

        if not isinstance(iface := self.cfg.get("iface"), str) or not iface:
            raise BarConfigError("'iface' must be specified")

        self.iface = iface

        self.qry_options = WifiOptions.NONE

        for fld in set(self.formatter.get_fields(self.content.label)):
            if fld not in self.FMT_FIELDS:
                raise BarConfigError(f"unknown label field: {fld}")
            self.qry_options |= WifiOptions.for_name(fld)

        if self.ramp:
            self.qry_options |= WifiOptions.SIGNAL

        if self.qry_options == WifiOptions.NONE:
            raise BarConfigError("no known format fields for label")

    def ramp_index(self, ramp_level: int) -> int | None:
        """The first ramp entry is shown when there is no signal, the rest
        is spread over the signal strength."""
        if not self.ramp:
            return None

        if ramp_level < 0 or len(self.ramp) == 1:
            return 0

        level = min(ramp_level, self.MAX_SIGNAL - 1)
        return int(level / (self.MAX_SIGNAL / (len(self.ramp) - 1))) + 1

    async def run(self):
        last_info = None

        while await self.sleep_interval():
            info = await anyio.to_thread.run_sync(
                self.backend.get_info, self.iface, self.qry_options
            )

            if info != last_info:
                last_info = info

                ramp_level = -1

                if info.percentage is not None and info.percentage >= 0:
                    ramp_level = info.percentage

                self.set_new_content_i(ramp_level, **asdict(info))
