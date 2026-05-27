from dataclasses import asdict

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase, WidgetContent

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
    ]

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        backend = self.cfg.get("backend")

        if backend not in self.BACKEND_MAP:
            raise BarConfigError(f"unknown backend: {backend}")

        self.dbus_iface = self.BACKEND_MAP[backend](None)

        self._last_info = None

        self.iface = self.cfg.get("iface")

        self.qry_options = WifiOptions.NONE

        for fld in set(self.formatter.get_fields(self.content.label)):
            if fld in self.FMT_FIELDS:
                opt = WifiOptions.for_name(fld)

                if opt is not WifiOptions.NONE:
                    self.qry_options |= opt
            else:
                raise BarConfigError(f"unknown label field: {fld}")

        if self.qry_options == WifiOptions.NONE:
            raise BarConfigError("no known format fields for label")

    def get_ramp(self, ramp_level: int = -1) -> WidgetContent | None:

        if ramp_level not in self.ramp_index_cache:
            ramp = self.cfg.get("ramp")
            content = None

            if ramp is not None and ramp:
                if ramp_level > -1:
                    level_ = min(ramp_level, self.MAX_SIGNAL - 1)
                    idx = int(level_ / (self.MAX_SIGNAL / (len(ramp) - 1))) + 1
                else:
                    idx = 0

                content = WidgetContent.parse(ramp[idx])
            self.ramp_index_cache[ramp_level] = content
        return self.ramp_index_cache[ramp_level]

    async def run(self):

        info = None

        while await self.sleep_interval():
            info = await self.dbus_iface.get_info(self.iface, self.qry_options)

            if not info.matches(self._last_info):
                self._last_info = info

                ramp_level = -1

                if info.percentage is not None and info.percentage >= 0:
                    ramp_level = info.percentage

                self.set_new_content_i(ramp_level=ramp_level, **asdict(info))
