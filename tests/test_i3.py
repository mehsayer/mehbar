import asyncio
import json
import socket
import struct

import pytest

pytest.importorskip("i3ipc")

from i3ipc._private import MessageType  # noqa: E402

from mehbar._widgets._i3 import I3Connection  # noqa: E402

MAGIC = b"i3-ipc"


def _header(length: int, msg_type: int) -> bytes:
    return MAGIC + struct.pack("=II", length, msg_type)


async def _client_and_server():
    client, server = socket.socketpair()
    client.setblocking(False)
    server.setblocking(False)

    conn = I3Connection()
    conn._loop = asyncio.get_running_loop()
    conn._cmd_socket = client
    conn._cmd_lock = asyncio.Lock()
    conn._broken = None

    return conn, server


async def _read_request(loop, server) -> int:
    header = await loop.sock_recv(server, len(MAGIC) + 8)
    length, msg_type = struct.unpack("=II", header[len(MAGIC) :])
    if length:
        await loop.sock_recv(server, length)
    return msg_type


def test_fragmented_reply_is_read_completely():
    async def _test():
        conn, server = await _client_and_server()
        loop = asyncio.get_running_loop()
        body = json.dumps({"nodes": ["x" * 1000] * 100}).encode()

        async def _serve():
            msg_type = await _read_request(loop, server)
            reply = _header(len(body), msg_type) + body
            # the reply arrives in pieces, as large ones do
            for start in range(0, len(reply), 4096):
                await loop.sock_sendall(server, reply[start : start + 4096])
                await asyncio.sleep(0.01)

        serving = asyncio.ensure_future(_serve())
        reply = await conn._message(MessageType.GET_TREE)
        await serving

        assert reply == body

    asyncio.run(_test())


def test_concurrent_requests_do_not_interleave():
    async def _test():
        conn, server = await _client_and_server()
        loop = asyncio.get_running_loop()

        async def _serve():
            for _ in range(2):
                msg_type = await _read_request(loop, server)
                body = str(msg_type).encode()
                await loop.sock_sendall(server, _header(len(body), msg_type) + body)

        serving = asyncio.ensure_future(_serve())
        replies = await asyncio.gather(
            conn._message(MessageType.GET_TREE),
            conn._message(MessageType.GET_WORKSPACES),
        )
        await serving

        assert replies == [
            str(MessageType.GET_TREE.value).encode(),
            str(MessageType.GET_WORKSPACES.value).encode(),
        ]

    asyncio.run(_test())


def test_timeout_breaks_the_connection(monkeypatch):
    monkeypatch.setattr(I3Connection, "REQUEST_TIMEOUT", 0.1)

    async def _test():
        conn, _server = await _client_and_server()

        # no reply
        with pytest.raises(ConnectionError):
            await conn._message(MessageType.GET_TREE)

        # a late reply would be taken for the reply to the next request
        with pytest.raises(ConnectionError, match="broken"):
            await conn._message(MessageType.GET_TREE)

    asyncio.run(_test())
