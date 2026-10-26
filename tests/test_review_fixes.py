"""Regression tests of the review of 2026-10-26."""

import logging
import threading

import anyio
import pytest
from gi.repository import Gio, GLib, Gtk


@pytest.mark.display
def test_hung_disk_read_blocks_one_thread_only(make_res_mgr, monkeypatch):
    from mehbar._widgets import disk

    release = threading.Event()
    calls = []

    def _hung(path):
        calls.append(path)
        release.wait(5)

    monkeypatch.setattr(disk, "disk_usage", _hung)
    monkeypatch.setattr(disk.WidgetDiskUsage, "TIMEOUT", 0.1)
    widget = disk.WidgetDiskUsage(
        "d", make_res_mgr('[start.d]\ntype = "disk_usage"\ninterval = 30')
    )

    async def _poll_twice():
        for _ in range(2):
            with pytest.raises(TimeoutError):
                await widget._disk_usage()

    try:
        anyio.run(_poll_twice)
        assert len(calls) == 1
    finally:
        release.set()


@pytest.mark.display
def test_exec_tail_lets_the_command_finish(make_res_mgr, caplog):
    from mehbar._widgets.exec import WidgetExecTail

    # closes its output, then exits after a while
    cmdline = "echo done; exec >&-; sleep 0.3; exit 3"
    widget = WidgetExecTail(
        "e",
        make_res_mgr(
            f'[start.e]\ntype = "exec_tail"\ncmdline = "{cmdline}"\nlabel = "{{label}}"'
        ),
    )

    with caplog.at_level(logging.WARNING):
        anyio.run(widget.run)

    # not terminated by the widget, which would be status -15
    assert "command exited with status 3" in caplog.text


@pytest.mark.display
def test_no_tooltip_popover_without_a_tooltip(make_res_mgr):
    from mehbar._widgets.datetime import WidgetDateTime
    from mehbar._widgets.static import WidgetStatic

    res_mgr = make_res_mgr(
        '[start.s]\ntype = "static"\nlabel = "x"\n'
        '[start.c]\ntype = "datetime"\ninterval = 60\n'
    )
    static, clock = WidgetStatic("s", res_mgr), WidgetDateTime("c", res_mgr)

    for widget in (static, clock):
        Gtk.Box().append(widget)
        widget._setup_tooltip()

    assert static._popover is None
    # the calendar
    assert clock._popover is not None


def test_dbus_facade_reconnects():
    from mehbar._widgets._dbus_facade import DBusFacade

    try:
        Gio.bus_get_sync(Gio.BusType.SESSION, None)
    except GLib.Error:
        pytest.skip("no session bus")

    class _Session(DBusFacade):
        BUS_TYPE = Gio.BusType.SESSION

    facade = _Session()
    sub_id = facade.owner_subscribe(lambda: None)
    old_bus = facade.bus
    # as when the bus daemon restarts; a private connection, the shared one
    # is used by the other tests
    facade._bus = Gio.DBusConnection.new_for_address_sync(
        Gio.dbus_address_get_for_bus_sync(Gio.BusType.SESSION, None),
        Gio.DBusConnectionFlags.AUTHENTICATION_CLIENT
        | Gio.DBusConnectionFlags.MESSAGE_BUS_CONNECTION,
        None,
        None,
    )
    facade._bus.close_sync(None)

    (names,) = facade.call(
        "/org/freedesktop/DBus", "org.freedesktop.DBus", "ListNames", None, "(as)"
    )

    assert names and not facade.bus.is_closed()
    facade.signal_unsubscribe(sub_id)
    assert not facade._subscriptions and old_bus is not None


def test_dropping_an_i3_connection_ends_its_watcher():
    from mehbar.resource_manager import ResourceManager

    class _Conn:
        quit = closed = False

        def main_quit(self):
            self.quit = True

        def close(self):
            self.closed = True

    async def _drop():
        res_mgr = ResourceManager.__new__(ResourceManager)
        conn, lost = _Conn(), anyio.Event()
        res_mgr._i3_connection = (conn, lost)
        res_mgr.drop_i3_connection(conn)
        return conn, lost

    conn, lost = anyio.run(_drop)

    assert conn.quit and conn.closed and lost.is_set()


def test_i3_socket_errors_are_connection_errors(monkeypatch):
    pytest.importorskip("i3ipc")
    import asyncio
    import socket

    from i3ipc._private import MessageType

    from mehbar._widgets._i3 import I3Connection

    async def _request():
        client, server = socket.socketpair()
        client.setblocking(False)
        conn = I3Connection()
        conn._loop = asyncio.get_running_loop()
        conn._cmd_socket = client
        conn._cmd_lock = asyncio.Lock()
        conn._broken = None
        client.close()

        with pytest.raises(ConnectionError):
            await conn._message(MessageType.GET_TREE)

    asyncio.run(_request())


def test_i3_main_quit_after_cancelled_main():
    pytest.importorskip("i3ipc")
    import asyncio

    from mehbar._widgets._i3 import I3Connection

    async def _quit():
        conn = I3Connection()
        conn._loop = asyncio.get_running_loop()
        main = asyncio.ensure_future(conn.main())
        await asyncio.sleep(0)
        main.cancel()
        await asyncio.sleep(0)
        # as when the bar exits
        conn.main_quit()

    asyncio.run(_quit())


@pytest.mark.parametrize(
    ("values", "delta", "limit", "expected"),
    [
        # each channel is limited, the louder one does not overshoot
        ([0.90, 0.96], 0.10, 1.0, [1.0, 1.0]),
        ([0.75, 0.75], 0.10, 0.8, [0.8, 0.8]),
        # raised above the limit elsewhere, not lowered by scrolling up
        ([1.30, 1.30], 0.10, 1.0, [1.30, 1.30]),
        ([1.30, 0.90], 0.10, 1.0, [1.30, 1.0]),
        ([0.05, 0.50], -0.10, 1.0, [0.0, 0.40]),
    ],
)
def test_change_volumes(values, delta, limit, expected):
    pytest.importorskip("pulsectl_asyncio")
    from mehbar._widgets.pulse_volume import change_volumes

    assert change_volumes(values, delta, limit) == pytest.approx(expected)
