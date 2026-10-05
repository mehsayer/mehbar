import enum
from collections.abc import Callable
from dataclasses import dataclass

from ._dbus_facade import DBusFacade


class BluetoothStatus(enum.IntEnum):
    OFF = 0
    ON = 1
    CONNECTED = 2


@dataclass(frozen=True)
class BluetoothInfo:
    status: BluetoothStatus
    device: str | None = None
    name: str | None = None
    alias: str | None = None
    icon: str | None = None
    bat_percent: int = -1
    volume: int = -1


class BluezBackend(DBusFacade):
    BASE_SVC = "org.bluez"

    ADAPTER_IFACE = "org.bluez.Adapter1"
    DEVICE_IFACE = "org.bluez.Device1"
    BATTERY_IFACE = "org.bluez.Battery1"
    MTRANS_IFACE = "org.bluez.MediaTransport1"

    # Device properties that change often but do not affect the status
    NOISY_PROPS = frozenset({"RSSI", "TxPower", "ManufacturerData", "ServiceData"})

    def get_info(self) -> BluetoothInfo:
        """Blocks, call it in a worker thread."""

        objects = self.get_managed_objects()

        is_powered = any(
            ifaces[self.ADAPTER_IFACE].get("Powered", False)
            for ifaces in objects.values()
            if self.ADAPTER_IFACE in ifaces
        )

        if not is_powered:
            return BluetoothInfo(BluetoothStatus.OFF)

        for path, ifaces in sorted(objects.items()):
            dev_props = ifaces.get(self.DEVICE_IFACE)

            if not dev_props or not dev_props.get("Connected", False):
                continue

            bat_percent = ifaces.get(self.BATTERY_IFACE, {}).get("Percentage", -1)

            volume = -1
            for ifaces_ in objects.values():
                mtrans = ifaces_.get(self.MTRANS_IFACE)
                if mtrans and mtrans.get("Device") == path:
                    volume = mtrans.get("Volume", -1)
                    break

            return BluetoothInfo(
                BluetoothStatus.CONNECTED,
                path,
                dev_props.get("Name"),
                dev_props.get("Alias"),
                dev_props.get("Icon"),
                bat_percent,
                volume,
            )

        return BluetoothInfo(BluetoothStatus.ON)

    def subscribe(self, callback: Callable[[], None]) -> list[int]:
        """Calls `callback` on GTK main thread whenever adapters or devices
        change. Returns subscription IDs."""

        def _on_props_changed(_path, _iface, _member, args):
            iface, changed, invalidated = args
            if iface == self.DEVICE_IFACE and not invalidated:
                if changed.keys() <= self.NOISY_PROPS:
                    return
            callback()

        def _on_ifaces_changed(*_):
            callback()

        sub_ids = [
            self.signal_subscribe(_on_ifaces_changed, self.OBJ_MANAGER_IFACE, member)
            for member in ("InterfacesAdded", "InterfacesRemoved")
        ]

        for iface in (
            self.ADAPTER_IFACE,
            self.DEVICE_IFACE,
            self.BATTERY_IFACE,
            self.MTRANS_IFACE,
        ):
            sub_ids.append(
                self.signal_subscribe(
                    _on_props_changed, self.PROPS_IFACE, "PropertiesChanged", arg0=iface
                )
            )

        return sub_ids

    def unsubscribe(self, sub_ids: list[int]):
        for sub_id in sub_ids:
            self.signal_unsubscribe(sub_id)
