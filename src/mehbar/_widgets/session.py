import getpass
import os
import socket
import time

from mehbar.resource_manager import ResourceManager
from mehbar.tools import FormattableTimeDelta
from mehbar.widget import WidgetBase


class WidgetSession(WidgetBase):
    TYPE = "session"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.username = getpass.getuser()
        self.uid = os.getuid()
        self.hostname = socket.gethostname()
        self.fqdn = socket.getfqdn()

    async def run(self):
        while await self.sleep_interval():
            uptime_sec = time.clock_gettime(time.CLOCK_BOOTTIME)

            self.set_new_content_i(
                username=self.username,
                uid=self.uid,
                hostname=self.hostname,
                fqdn=self.fqdn,
                uptime=FormattableTimeDelta(uptime_sec),
            )
