import os
import getpass
import time
from mehbar.widgets import Widget
from mehbar.tools import FormattableTimeDelta
from datetime import timedelta, datetime
import socket
import time

class WidgetSession(Widget):
    def __init__(self, interval: int, label_format: str):
        super().__init__(interval, label_format)

        self.username = getpass.getuser()
        self.uid = os.getuid()
        self.hostname = socket.gethostname()
        self.fqdn = socket.getfqdn()


    def update(self):
        uptime_sec = time.clock_gettime(time.CLOCK_BOOTTIME)

        self.format_label_idle(username=self.username,
                               uid=self.uid,
                               hostname=self.hostname,
                               fqdn=self.fqdn,
                               uptime=FormattableTimeDelta(uptime_sec))
