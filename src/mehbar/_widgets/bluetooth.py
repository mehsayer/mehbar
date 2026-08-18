import logging

from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase

from ._bluetooth import BluetoothEvent, BluetoothStatus, BluezBackend


class WidgetBluetoothStatus(WidgetBase):
    TYPE = "bluetooth-status"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self._last_info = None
        self.max_ramp_level = self.cfg.get("max_ramp_level", self.DFL_RAMP_IDX)
        self.dbus_iface = BluezBackend(BluetoothEvent.POWER, self._cb)

    def _cb(self, *args, **kwargs):
        logging.debug("TOP CALLBACK ARGS=%s; KWARGS=%s", args, kwargs)

    async def run(self):

        await self.dbus_iface.start()

        # while await self.sleep_interval():
        #     if (info := await self.dbus_iface.get_info()) != self._last_info:
        #         self._last_state = info

        #         ramp_level = int(
        #             (info.status + 1) * (self.max_ramp_level / len(BluetoothStatus))
        #         )

        #         self.set_new_content_i(ramp_level=ramp_level)
