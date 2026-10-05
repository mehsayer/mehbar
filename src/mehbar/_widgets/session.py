import getpass
import os
import socket
import time

import anyio

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

    async def run(self):
        # may take a while if DNS is slow
        fqdn = await anyio.to_thread.run_sync(socket.getfqdn)

        while await self.sleep_interval():
            uptime_sec = time.clock_gettime(time.CLOCK_BOOTTIME)

            self.set_new_content_i(
                username=self.username,
                uid=self.uid,
                hostname=self.hostname,
                fqdn=fqdn,
                uptime=FormattableTimeDelta(uptime_sec),
            )
