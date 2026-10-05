from operator import itemgetter

import anyio

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


class WidgetNetworkRate(WidgetBase):
    """Shows transfer rates of the `iface` network interface, or all
    interfaces. `conv_map` maps unit names to their size in bytes."""

    DEFAULT_CONVERSIONS = {"Kb/s": 1024, "Mb/s": 1024**2, "b/s": 1}
    UNIQUE = False
    TYPE = "network_rate"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        conv_map = self.cfg.get("conv_map") or self.DEFAULT_CONVERSIONS

        if not isinstance(conv_map, dict) or not all(
            isinstance(div, (int, float)) and div > 0 for div in conv_map.values()
        ):
            raise BarConfigError("'conv_map' must map unit names to positive numbers")

        self.conv_map = sorted(conv_map.items(), key=itemgetter(1), reverse=True)

        iface = self.cfg.get("iface")
        self.iface = None if iface in (None, "all") else iface

    def _conv_rate(self, rate_bytes: int) -> tuple[int, str]:
        for unit, divisor in self.conv_map:
            if (value := int(rate_bytes // divisor)) > 0:
                return value, unit

        return rate_bytes, self.conv_map[-1][0]

    def _read(self):
        import psutil

        if self.iface is None:
            return psutil.net_io_counters()
        return psutil.net_io_counters(pernic=True).get(self.iface)

    async def run(self):
        last_info = None
        last_ts = 0.0

        while await self.sleep_interval():
            info = self._read()
            now = anyio.current_time()

            tx_rate = 0
            rx_rate = 0

            if info is not None and last_info is not None and now > last_ts:
                t_delta = now - last_ts
                # counters are reset when the interface is recreated
                tx_rate = max(0, int((info.bytes_sent - last_info.bytes_sent) / t_delta))
                rx_rate = max(0, int((info.bytes_recv - last_info.bytes_recv) / t_delta))

            last_info = info
            last_ts = now

            rate_tx, unit_tx = self._conv_rate(tx_rate)
            rate_rx, unit_rx = self._conv_rate(rx_rate)

            self.set_new_content_i(
                rate_tx=rate_tx,
                unit_tx=unit_tx,
                rate_rx=rate_rx,
                unit_rx=unit_rx,
            )
