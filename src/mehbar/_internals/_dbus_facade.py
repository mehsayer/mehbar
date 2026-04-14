import asyncio
from functools import partial
from typing import Any

from gi.repository import Gio, GLib


class DBusFacade:
    FLAGS_NOSIG = (
        Gio.DBusProxyFlags.DO_NOT_CONNECT_SIGNALS | Gio.DBusProxyFlags.DO_NOT_AUTO_START
    )
    FLAGS_NOSIG_NOPROP = (
        Gio.DBusProxyFlags.DO_NOT_CONNECT_SIGNALS
        | Gio.DBusProxyFlags.DO_NOT_AUTO_START
        | Gio.DBusProxyFlags.DO_NOT_LOAD_PROPERTIES
    )
    CALL_TIMEOUT_MS = 500

    DBUS_IFACE = "org.freedesktop.DBus"
    DBUS_ROOT_OBJ = "/org/freedesktop/DBus"
    BASE_SVC = "org.freedesktop.DBus"

    def __init__(self, bus: Gio.DBusConnection | None, svc: str | None = None):
        if bus is None:
            self.bus = Gio.bus_get_sync(Gio.BusType.SYSTEM, None)
        else:
            self.bus = bus

        if svc is None:
            self.svc = self.BASE_SVC
        else:
            self.svc = svc

        self._ensure_available()

    def _ensure_available(self):
        proxy = Gio.DBusProxy.new_sync(
            self.bus,
            self.FLAGS_NOSIG_NOPROP,
            None,
            self.DBUS_IFACE,
            self.DBUS_ROOT_OBJ,
            self.DBUS_IFACE,
            None,
        )

        result = proxy.call_sync(
            "ListNames",
            None,
            Gio.DBusCallFlags.NO_AUTO_START,
            self.CALL_TIMEOUT_MS,
            None,
        )

        if self.svc not in result.unpack()[0]:
            raise RuntimeError(f"name {self.svc} not available")

    def p_new(
        self,
        iface: str,
        obj: str | GLib.Variant,
        flags: Gio.DBusCallFlags | None = None,
    ) -> Gio.DBusProxy:

        if isinstance(obj, GLib.Variant):
            obj = obj.get_string()

        _flags = self.FLAGS_NOSIG

        if flags is not None:
            _flags |= flags

        return Gio.DBusProxy.new_sync(
            self.bus, _flags, None, self.svc, obj, iface, None
        )

    async def p_new_async(
        self,
        iface: str,
        obj: str | GLib.Variant,
        flags: Gio.DBusCallFlags | None = None,
    ) -> Gio.DBusProxy:

        if isinstance(obj, GLib.Variant):
            obj = obj.get_string()

        _flags = self.FLAGS_NOSIG

        if flags is not None:
            _flags |= flags

        proxy_ready_event = asyncio.Event()
        proxy_ready_event.clear()
        proxy_exception = None
        proxy = None

        def _proxy_ready_cb(proxy_obj, result, *_):
            try:
                nonlocal proxy
                proxy = proxy_obj.new_finish(result)
            except GLib.Error as ex:
                nonlocal proxy_exception
                proxy_exception = ex
            finally:
                proxy_ready_event.set()

        aio_loop = asyncio.get_running_loop()

        # Note, that the callback is run on main thread, so we need to
        # schedule it to run on the thread running asyncio loop
        Gio.DBusProxy.new(
            self.bus,
            _flags,
            None,
            self.svc,
            obj,
            iface,
            None,
            partial(aio_loop.call_soon_threadsafe, _proxy_ready_cb),
        )

        await proxy_ready_event.wait()

        if proxy_exception is not None:
            raise proxy_exception

        return proxy

    def p_new_call(
        self, proxy: Gio.DBusProxy, method: str, arg: GLib.Variant | str | None = None
    ) -> Any:

        if arg is not None and isinstance(arg, str):
            arg = GLib.Variant("(s)", (arg,))

        ret = proxy.call_sync(
            method, arg, Gio.DBusCallFlags.NO_AUTO_START, self.CALL_TIMEOUT_MS, None
        )

        if ret is not None:
            ret = ret.unpack()[0]

        return ret

    async def p_new_call_async(
        self, proxy: Gio.DBusProxy, method: str, arg: GLib.Variant | str | None = None
    ) -> Any:

        if arg is not None and isinstance(arg, str):
            arg = GLib.Variant("(s)", (arg,))

        call_ready_event = asyncio.Event()
        call_result = None
        call_exception = None

        def _call_ready_cb(proxy_obj, result, *_):
            try:
                nonlocal call_result
                call_result = proxy_obj.call_finish(result)
            except GLib.Error as ex:
                nonlocal call_exception
                call_exception = ex
            finally:
                call_ready_event.set()

        aio_loop = asyncio.get_running_loop()

        proxy.call(
            method,
            arg,
            Gio.DBusCallFlags.NO_AUTO_START,
            self.CALL_TIMEOUT_MS,
            None,
            partial(aio_loop.call_soon_threadsafe, _call_ready_cb),
        )

        await call_ready_event.wait()

        if call_exception is not None:
            raise call_exception

        ret = None

        if call_result is not None:
            ret = call_result.unpack()[0]

        return ret

    def new_call(
        self,
        iface: str,
        obj: str | GLib.Variant,
        method: str,
        arg: GLib.Variant | str | None = None,
    ) -> Any:
        proxy = self.p_new(iface, obj)
        return self.p_new_call(proxy, method, arg)

    async def new_call_async(
        self,
        iface: str,
        obj: str | GLib.Variant,
        method: str,
        arg: GLib.Variant | str | None = None,
    ) -> Any:
        proxy = await self.p_new_async(iface, obj)
        return await self.p_new_call_async(proxy, method, arg)

    def p_get_prop(self, proxy: Gio.DBusProxy, prop: str) -> Any:
        ret = None
        if (_ret := proxy.get_cached_property(prop)) is not None:
            if isinstance(_ret, GLib.Variant):
                ret = _ret.unpack()
            else:
                ret = _ret
        return ret

    def get_prop(self, iface: str, obj: GLib.Variant | str, prop: str) -> str:
        ret = None

        if obj is not None:
            proxy = self.p_new(iface, obj)
            ret = self.p_get_prop(proxy, prop)
        return ret

    async def get_prop_async(
        self, iface: str, obj: GLib.Variant | str, prop: str
    ) -> str:
        ret = None

        if obj is not None:
            proxy = await self.p_new_async(iface, obj)
            ret = self.p_get_prop(proxy, prop)
        return ret
