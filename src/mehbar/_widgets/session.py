import getpass
import os
import socket
import time

from mehbar.tools import FormattableTimeDelta
from mehbar.widget import WidgetBase


class WidgetSession(WidgetBase):
    TYPE = "session"

    def __init__(self, interval: int, label_format: str):
        super().__init__(interval, label_format)

        self.username = getpass.getuser()
        self.uid = os.getuid()
        self.hostname = socket.gethostname()
        self.fqdn = socket.getfqdn()

    async def run(self):
        while await self.sleep_interval():
            uptime_sec = time.clock_gettime(time.CLOCK_BOOTTIME)

            self.format_label_idle(
                username=self.username,
                uid=self.uid,
                hostname=self.hostname,
                fqdn=self.fqdn,
                uptime=FormattableTimeDelta(uptime_sec),
            )
