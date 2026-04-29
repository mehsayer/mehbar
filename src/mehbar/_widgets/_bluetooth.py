import enum

from mehbar._internals import DBusFacade
from mehbar.exceptions import CapabilityError
from mehbar.widgets import Widget


class BluetoothState(enum.IntEnum):
    OFF = 0
    ON = 1
    CONNECTED = 2


class BluezBackend(DBusFacade):
    BASE_SVC = "org.bluez"
    BASE_OBJ = "/"
    BASE_IFACE = "org.freedesktop.DBus.ObjectManager"

    ADAPT_OBJ_BASE = "org.bluez.Adapter"
    DEV_OBJ_BASE = "org.bluez.Device"

    async def get_state(self) -> BluetoothState:
        objects = await self.new_call_async(
            self.BASE_IFACE, self.BASE_OBJ, "GetManagedObjects"
        )

        if objects is None or not objects:
            raise CapabilityError("no 'bluez' managed objects found")

        is_powered = False
        is_connected = False

        for obj_map in objects.values():
            for obj_name, obj_props in obj_map.items():
                if not is_powered and obj_name.startswith(self.ADAPT_OBJ_BASE):
                    is_powered = obj_props.get("Powered", False)

                if not is_connected and obj_name.startswith(self.DEV_OBJ_BASE):
                    is_connected = obj_props.get("Connected", False)

        ret = BluetoothState.OFF

        if is_connected:
            ret = BluetoothState.CONNECTED
        elif is_powered:
            ret = BluetoothState.ON

        return ret


class WidgetBluetooth(Widget):
    def __init__(self, interval: int, label_format: str, ramp: list[str] | None = None):
        super().__init__(interval, label_format, ramp)

        self.dbus_iface = BluezBackend(None)

        self.ramps = []

        if self.ramp is not None:
            ramp_len = len(self.ramp)
            if ramp_len >= 3:
                self.ramps = self.ramp[:3]
            elif ramp_len > 0:
                self.ramps = self.ramp + ([""] * (3 - ramp_len))

    async def run(self):

        while await self.sleep_interval():
            state = await self.dbus_iface.get_state()

            if state != self._last_value:
                self._last_value = state

                ramp = None

                if self.ramps:
                    ramp = self.ramps[state]

                self.format_label_idle(ramp=ramp)
