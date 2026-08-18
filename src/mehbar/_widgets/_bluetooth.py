import enum
import json
import logging

# import time
from dataclasses import dataclass
from functools import partial
from typing import Callable

import anyio
from gi.repository import Gio, GLib

from mehbar import tools
from mehbar.exceptions import CapabilityError

from ._dbus_facade import DBusFacade


class BluetoothStatus(enum.IntEnum):
    OFF = 0
    ON = 1
    CONNECTED = 2
    UNKNOWN = 3


class BluetoothEvent(enum.IntEnum):
    POWER = 0
    ADDED = 1
    REMOVED = 2
    VOLUME = 3
    BATTERY = 4
    NAME = 5


@dataclass
class BluetoothInfo:
    device: str
    status: BluetoothStatus
    name: str | None
    alias: str | None
    icon: str | None
    bat_percent: int = -1
    volume: int = -1


class BluezBackend(DBusFacade, metaclass=tools.Singleton):
    BASE_SVC = "org.bluez"
    BASE_OBJ = "/"
    BASE_IFACE = "org.freedesktop.DBus.ObjectManager"

    ADAPT_OBJ = "org.bluez.Adapter1"
    DEV_OBJ = "org.bluez.Device1"
    BAT_OBJ = "org.bluez.Battery1"
    MTRANS_OBJ = "org.bluez.MediaTransport1"

    def __init__(self, events: BluetoothEvent, callback: Callable[[BluetoothInfo]]):
        super().__init__(None, self.BASE_SVC)
        self._sub_ids = set()
        self._info = None
        self._loop_token = None

        self._callback = callback

    def _dispatch(self):
        if self._info is not None and self._callback is not None:
            self._callback(self._info)

    async def _cb_added(self, path: str, *objects):

        if self._info is not None:
            for obj in objects:
                if len(obj) == 2:
                    node_name, obj_map = obj

                    if self._info.device is None or node_name.startswith(
                        self._info.device
                    ):
                        for obj_name, obj_props in obj_map.items():
                            if obj_name == self.MTRANS_OBJ:
                                self._info.volume = obj_props.get("Volume", -1)
                            elif obj_name == self.BAT_OBJ:
                                self._info.bat_percent = obj_props.get("Percentage", -1)
                            elif obj_name == self.DEV_OBJ:
                                # is_connected = obj_props.get("Connected", False)
                                #
                                if self._info.device is None:
                                    self._info.device = node_name

                                self._info.alias = obj_props.get("Alias")
                                self._info.name = obj_props.get("Name")
                                self._info.icon = obj_props.get("Icon")
                        logging.debug("UPDATED %s", self._info.device)

    async def _cb_removed(self, path: str, *objects):

        if self._info is not None:
            for obj in objects:
                if len(obj) == 2:
                    node_name, iface_list = obj

                    if self._info.device is not None and node_name.startswith(
                        self._info.device
                    ):
                        for obj_name in iface_list:
                            if obj_name == self.MTRANS_OBJ:
                                self._info.volume = -1
                            elif obj_name == self.BAT_OBJ:
                                self._info.bat_percent = -1
                            elif obj_name == self.DEV_OBJ:
                                # is_connected = obj_props.get("Connected", False)
                                #
                                self._info.device = None

                                self._info.alias = None
                                self._info.name = None
                                self._info.icon = None

        # logging.debug("BOTTOM CB ADDED path=<%s>; objects=%s", path, objects)

    async def _cb_power(self, *args, **kwargs):
        logging.debug("ADDED args=%s; kwargs=%s", args, kwargs)

    async def _cb_battery(self, *args, **kwargs):
        logging.debug("ADDED args=%s; kwargs=%s", args, kwargs)

    async def _cb_volume(self, *args, **kwargs):
        logging.debug("ADDED args=%s; kwargs=%s", args, kwargs)

    # def async_callback(self, callback_coro: Callable[[...]]):
    #     pass

    def _subscribe(self, device: str | None = None):

        self.disconnect_signals()

        cb_added = partial(
            anyio.from_thread.run, self._cb_added, token=self._loop_token
        )

        self._sub_ids.add(
            self.signal_subscribe(
                self.BASE_SVC,
                self.BASE_IFACE,
                "InterfacesAdded",
                None,
                None,
                Gio.DBusSignalFlags.NONE,
                cb_added,
            )
        )

        cb_removed = partial(
            anyio.from_thread.run, self._cb_removed, token=self._loop_token
        )

        self._sub_ids.add(
            self.signal_subscribe(
                self.BASE_SVC,
                self.BASE_IFACE,
                "InterfacesRemoved",
                None,
                None,
                Gio.DBusSignalFlags.NONE,
                cb_removed,
            )
        )

    def disconnect_signals(self):
        for sub_id in self._sub_ids:
            try:
                self.signal_unsubscribe(sub_id)
            except Glib.Error:
                logging.debug("cannot unsubscribe from ID %d", sub_id)
            logging.debug("DESTROYED")

    def __del__(self):
        self.disconnect_signals()

    async def start(self):

        self._loop_token = anyio.lowlevel.current_token()  # type: ignore

        self._info = await self.get_info()
        self._dispatch()
        self._subscribe()

    async def get_info(self):

        objects = await self.new_call_async(
            self.BASE_IFACE, self.BASE_OBJ, "GetManagedObjects"
        )

        if objects is None or not objects:
            raise CapabilityError("no BlueZ-managed objects found")

        is_powered = False
        is_connected = False
        name = None
        alias = None
        device = None
        icon = None
        bat_percent = -1
        volume = -1

        for node_name, obj_map in objects.items():
            for obj_name, obj_props in obj_map.items():
                if not is_powered and obj_name == self.ADAPT_OBJ:
                    is_powered = obj_props.get("Powered", False)

                if bat_percent < 0 and obj_name == self.BAT_OBJ:
                    bat_percent = obj_props.get("Percentage", -1)

                if device is None and obj_name == self.DEV_OBJ:
                    is_connected = obj_props.get("Connected", False)
                    alias = obj_props.get("Alias")
                    name = obj_props.get("Name")
                    icon = obj_props.get("Icon")
                    device = str(node_name)

                if volume < 0 and obj_name == self.MTRANS_OBJ:
                    if device is not None and obj_props.get("Device") == device:
                        volume = obj_props.get("Volume", -1)

        status = BluetoothStatus.OFF

        if is_connected:
            status = BluetoothStatus.CONNECTED
        elif is_powered:
            status = BluetoothStatus.ON

        return BluetoothInfo(device, status, name, alias, icon, bat_percent, volume)
