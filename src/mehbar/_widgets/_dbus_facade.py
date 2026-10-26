from collections.abc import Callable
from typing import Any

from gi.repository import Gio, GLib  # type: ignore


class DBusUnavailable(Exception):
    """The service, or an object of it, is not available at the moment, e.g.
    the service is restarting or the device was removed."""


# errors of the services and objects not being available at the moment
_UNAVAILABLE_DBUS_ERRORS = frozenset(
    {
        Gio.DBusError.SERVICE_UNKNOWN,
        Gio.DBusError.NAME_HAS_NO_OWNER,
        Gio.DBusError.NO_REPLY,
        Gio.DBusError.TIMEOUT,
        Gio.DBusError.TIMED_OUT,
        Gio.DBusError.DISCONNECTED,
        Gio.DBusError.NO_SERVER,
        Gio.DBusError.UNKNOWN_OBJECT,
        Gio.DBusError.UNKNOWN_INTERFACE,
    }
)
_UNAVAILABLE_IO_ERRORS = frozenset(
    {
        Gio.IOErrorEnum.TIMED_OUT,
        Gio.IOErrorEnum.CLOSED,
        Gio.IOErrorEnum.CONNECTION_CLOSED,
    }
)


class DBusFacade:
    """Thin synchronous D-Bus client for a single service.

    Calls block, run them in a worker thread, e.g. with
    `anyio.to_thread.run_sync()`. Signal callbacks are invoked on GTK main
    thread. Calls raise `DBusUnavailable` if the service or the object is
    not available at the moment."""

    BUS_TYPE = Gio.BusType.SYSTEM
    BASE_SVC = "org.freedesktop.DBus"
    CALL_TIMEOUT_MS = 1000

    PROPS_IFACE = "org.freedesktop.DBus.Properties"
    OBJ_MANAGER_IFACE = "org.freedesktop.DBus.ObjectManager"

    # service specific error names meaning that an object is not available
    UNAVAILABLE_ERROR_NAMES: frozenset[str] = frozenset()

    def __init__(self, bus: Gio.DBusConnection | None = None, svc: str | None = None):
        self._bus = bus
        self.svc = svc or self.BASE_SVC

    @property
    def bus(self) -> Gio.DBusConnection:
        if self._bus is None:
            self._bus = Gio.bus_get_sync(self.BUS_TYPE, None)
            # the connection is shared, by default the process gets SIGTERM
            # once it is closed, e.g. when the bus daemon restarts
            self._bus.set_exit_on_close(False)
        return self._bus

    def is_unavailable(self, ex: GLib.Error) -> bool:
        if ex.domain == GLib.quark_to_string(Gio.DBusError.quark()):
            return ex.code in _UNAVAILABLE_DBUS_ERRORS
        if ex.domain == GLib.quark_to_string(Gio.io_error_quark()):
            if ex.code == Gio.IOErrorEnum.DBUS_ERROR:
                return (
                    Gio.DBusError.get_remote_error(ex) in self.UNAVAILABLE_ERROR_NAMES
                )
            return ex.code in _UNAVAILABLE_IO_ERRORS
        return False

    def call(
        self,
        obj: str,
        iface: str,
        method: str,
        args: GLib.Variant | None = None,
        reply_type: str | None = None,
    ) -> tuple:
        """Calls the method, returns the unpacked reply tuple."""
        try:
            reply = self.bus.call_sync(
                self.svc,
                obj,
                iface,
                method,
                args,
                GLib.VariantType.new(reply_type) if reply_type else None,
                Gio.DBusCallFlags.NO_AUTO_START,
                self.CALL_TIMEOUT_MS,
                None,
            )
        except GLib.Error as ex:
            if self.is_unavailable(ex):
                raise DBusUnavailable(
                    f"{self.svc} {obj} {iface}.{method}: {ex.message}"
                ) from ex
            raise
        return reply.unpack() if reply is not None else ()

    def get_prop(self, obj: str, iface: str, prop: str) -> Any:
        (value,) = self.call(
            obj, self.PROPS_IFACE, "Get", GLib.Variant("(ss)", (iface, prop)), "(v)"
        )
        return value

    def get_all_props(self, obj: str, iface: str) -> dict[str, Any]:
        (props,) = self.call(
            obj, self.PROPS_IFACE, "GetAll", GLib.Variant("(s)", (iface,)), "(a{sv})"
        )
        return props

    def get_managed_objects(self, obj: str = "/") -> dict[str, dict[str, dict]]:
        (objects,) = self.call(
            obj, self.OBJ_MANAGER_IFACE, "GetManagedObjects", None, "(a{oa{sa{sv}}})"
        )
        return objects

    def signal_subscribe(
        self,
        callback: Callable[[str, str, str, Any], None],
        iface: str | None = None,
        member: str | None = None,
        obj: str | None = None,
        arg0: str | None = None,
    ) -> int:
        """`callback(path, iface, member, args)` is invoked on GTK main thread."""

        def _on_signal(_conn, _sender, path, iface_, member_, params):
            callback(path, iface_, member_, params.unpack())

        return self.bus.signal_subscribe(
            self.svc,
            iface,
            member,
            obj,
            arg0,
            Gio.DBusSignalFlags.NONE,
            _on_signal,
        )

    def owner_subscribe(self, callback: Callable[[], None]) -> int:
        """`callback()` is invoked on GTK main thread when the service
        appears or disappears, e.g. restarts."""

        def _on_signal(*_):
            callback()

        # subclasses override BASE_SVC
        return self.bus.signal_subscribe(
            DBusFacade.BASE_SVC,
            "org.freedesktop.DBus",
            "NameOwnerChanged",
            "/org/freedesktop/DBus",
            self.svc,
            Gio.DBusSignalFlags.NONE,
            _on_signal,
        )

    def signal_unsubscribe(self, sub_id: int):
        self.bus.signal_unsubscribe(sub_id)


def is_null_path(path: str | None) -> bool:
    return not path or path == "/"
