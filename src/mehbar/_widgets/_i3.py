import asyncio
import inspect
import socket
from collections.abc import Callable
from typing import Any

import anyio
from i3ipc import Event
from i3ipc._private import MessageType
from i3ipc.aio import Connection
from i3ipc.aio.connection import _MAGIC, _pack, _struct_header_size, _unpack_header

# raised when the connection to i3/sway is lost
CONNECTION_ERRORS = (OSError, EOFError)


class _WaitAllSocket:
    """Makes `recv(n)` of a blocking socket return all `n` bytes, unless the
    connection is closed. i3ipc reads each event with a single `recv()`,
    which may return only a part of a large event."""

    def __init__(self, sock: socket.socket):
        self._sock = sock

    def recv(self, size: int) -> bytes:
        return self._sock.recv(size, socket.MSG_WAITALL)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sock, name)


class I3Connection(Connection):
    """i3ipc.aio connection with fixed requests. i3ipc 2.2.1 sends requests
    on a blocking socket, which stalls the event loop until i3/sway replies,
    reads replies with a single `recv()`, which may return a partial reply,
    and lets concurrent requests interleave. A connection that fails a
    request is unusable, `main()` raises then."""

    REQUEST_TIMEOUT = 5

    async def connect(self) -> "I3Connection":
        await super().connect()

        self._sub_socket = _WaitAllSocket(self._sub_socket)
        self._cmd_socket.setblocking(False)
        self._cmd_lock = asyncio.Lock()
        self._broken: BaseException | None = None

        return self

    async def _recv_exactly(self, size: int) -> bytes:
        buf = bytearray()

        while len(buf) < size:
            if not (
                chunk := await self._loop.sock_recv(self._cmd_socket, size - len(buf))
            ):
                raise EOFError("connection closed by i3/sway")
            buf += chunk

        return bytes(buf)

    async def _message(self, message_type: MessageType, payload: str = "") -> bytes:
        if message_type is MessageType.SUBSCRIBE:
            raise ValueError("cannot subscribe on the command socket")

        async with self._cmd_lock:
            if self._broken is not None:
                raise ConnectionError("i3/sway connection is broken") from self._broken

            try:
                async with asyncio.timeout(self.REQUEST_TIMEOUT):
                    await self._loop.sock_sendall(
                        self._cmd_socket, _pack(message_type, payload)
                    )
                    magic, length, reply_type = _unpack_header(
                        await self._recv_exactly(_struct_header_size)
                    )

                    if magic != _MAGIC or reply_type != message_type.value:
                        raise ConnectionError("unexpected reply from i3/sway")

                    return await self._recv_exactly(length)
            except BaseException as ex:
                # e.g. a timeout or a cancellation: the reply would be read
                # as the reply to the next request
                self._broken = ex
                self.main_quit(_error=ConnectionError("i3/sway request failed"))

                if isinstance(ex, TimeoutError):
                    raise ConnectionError("i3/sway did not reply") from ex
                raise

    def close(self):
        try:
            self._loop.remove_reader(self._sub_fd)
        except (
            AttributeError,
            ValueError,
            RuntimeError,
        ):  # never connected, or closed loop
            pass

        for sock in (
            getattr(self, "_cmd_socket", None),
            getattr(self, "_sub_socket", None),
        ):
            if sock is not None:
                sock.close()


class I3WidgetMixin:
    """Runs `i3_start()` whenever connected to i3/sway IPC, and reconnects
    when the connection is lost, e.g. when i3 restarts. Handlers added with
    `i3_on()` are removed on reconnect. An exception raised by a handler
    disables the widget, as one raised by `run()` does."""

    I3_RECONNECT_DELAY = 1.0

    async def i3_start(self, conn: I3Connection):
        """Adds event handlers, then shows the current state."""
        raise NotImplementedError()

    def i3_on(self, conn: I3Connection, event: Event, handler: Callable[[Any], Any]):
        """`handler(event)` may be a coroutine function."""
        handlers = self._i3_handlers

        async def _handler(_conn, event_):
            try:
                if inspect.isawaitable(result := handler(event_)):
                    await result
            except CONNECTION_ERRORS as ex:
                self.res_mgr.drop_i3_connection(conn, ex)  # type: ignore[attr-defined]
            except Exception as ex:
                if self._i3_error is None:
                    self._i3_error = ex
                self._i3_wake.set()

        conn.on(event, _handler)
        handlers.append(_handler)

    async def run(self):
        res_mgr = self.res_mgr  # type: ignore[attr-defined]
        connected = False

        while True:
            try:
                conn, lost = await res_mgr.get_i3_connection_async()
            except Exception:
                # i3/sway is not running at all
                if not connected:
                    raise
                await anyio.sleep(self.I3_RECONNECT_DELAY)
                continue

            connected = True
            self._i3_handlers: list[Callable] = []
            self._i3_error: Exception | None = None
            self._i3_wake = anyio.Event()

            try:
                async with anyio.create_task_group() as grp:

                    async def _wait_lost(lost_: anyio.Event):
                        await lost_.wait()
                        self._i3_wake.set()

                    grp.start_soon(_wait_lost, lost)

                    await self.i3_start(conn)
                    await self._i3_wake.wait()
                    grp.cancel_scope.cancel()
            except* CONNECTION_ERRORS as group:
                # the connection broke during a request
                res_mgr.drop_i3_connection(conn, group.exceptions[0])
            finally:
                for handler in self._i3_handlers:
                    conn.off(handler)

            if self._i3_error is not None:
                raise self._i3_error

            await anyio.sleep(self.I3_RECONNECT_DELAY)
