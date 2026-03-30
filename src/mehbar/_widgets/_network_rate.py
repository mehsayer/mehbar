import psutil
from mehbar.widgets import BarWidget
from operator import itemgetter
import time

class BarWidgetNetworkRate(BarWidget):

    def __init__(self, interval: int, iface: str, label_format: str, conv_map: dict[int, str]):
        super().__init__(interval, label_format)

        self.conv_map = sorted(conv_map.items(), key=itemgetter(1), reverse=True)
        self.iface = None

        self.rate_ts = time.monotonic()
        self.bytes_sent = 0
        self.bytes_recv = 0

        if iface is not None and iface != 'all':
            self.iface = iface


    def _conv_rate(self, rate_bytes: int):

        ret = None

        for unit, divisor in self.conv_map:
            if (value := rate_bytes // divisor) > 0:
                ret = (value, unit)
                break

        if ret is None:
            ret = (rate_bytes, self.conv_map[-1][0])

        return ret

    def update(self):

        rx_rate = 0
        tx_rate = 0

        if self.iface is None:
            info = psutil.net_io_counters()
        else:
            info = psutil.net_io_counters(pernic=True).get(self.iface)

        if info is not None:

            t_now = time.monotonic()

            t_delta = t_now - self.rate_ts
            self.rate_ts = t_now

            tx_delta = info.bytes_sent - self.bytes_sent
            self.bytes_sent = info.bytes_sent

            rx_delta = info.bytes_recv - self.bytes_recv
            self.bytes_recv = info.bytes_recv

            if t_delta > 0:
                tx_rate = int(tx_delta / t_delta)
                rx_rate = int(rx_delta / t_delta)

            comp_rate = complex(tx_rate, rx_rate)

            if self._last_value != comp_rate:
                self._last_value = comp_rate

                res_tx = self._conv_rate(tx_rate)
                res_rx = self._conv_rate(rx_rate)

                self.format_label_idle(rate_tx=res_tx[0], unit_tx=res_tx[1], rate_rx=res_rx[0], unit_rx=res_rx[1])



