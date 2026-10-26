import subprocess
import time
from pathlib import Path

import anyio
import pytest

from mehbar._widgets.exec import iter_lines, open_command, stop_command
from mehbar._widgets.file import read_last_line


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"first\nsecond\n", "second"),
        (b"first\nsecond", "second"),
        (b"first\nsecond\n\n  \n", "second"),
        (b"only", "only"),
        (b"", None),
        (b"\n\n", None),
    ],
)
def test_read_last_line(tmp_path, data, expected):
    path = tmp_path / "f"
    path.write_bytes(data)
    assert read_last_line(path, chunk_size=4) == expected


def test_read_last_line_of_a_long_line_is_bounded(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"x" * (8 << 20))

    started = time.monotonic()
    line = read_last_line(path, max_len=1024)

    assert line == "x" * 1024
    assert time.monotonic() - started < 1


def test_read_last_line_of_procfs_files():
    # procfs files report no size, /proc/self/status has many lines
    assert read_last_line(Path("/proc/loadavg"))

    path = Path("/proc/self/status")
    # the last line depends on the kernel version
    assert read_last_line(path) == path.read_text().rstrip().splitlines()[-1]


class _Chunks:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


async def _lines(chunks, max_len):
    return [line async for line in iter_lines(_Chunks(chunks), max_len)]


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        ([b"a\nb", b"c\n"], ["a", "bc"]),
        ([b"a\n", b"tail"], ["a", "tail"]),
        # the whole line that is too long is dropped, not only its start
        ([b"HEADxxxx", b"TAIL\nok\n"], ["ok"]),
        ([b"HEADxxxx"], []),
    ],
)
def test_iter_lines(chunks, expected):
    assert anyio.run(_lines, chunks, 4) == expected


def _group_alive(pgid: int) -> bool:
    out = subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True)
    return bool(out.stdout.strip())


def test_stop_command_stops_the_whole_pipeline():
    async def _run():
        with anyio.move_on_after(0.5):
            async with await open_command("sleep 30 | sleep 31") as proc:
                try:
                    await proc.wait()
                finally:
                    await stop_command(proc)
        return proc

    proc = anyio.run(_run)

    assert proc.returncode is not None
    assert not _group_alive(proc.pid)
