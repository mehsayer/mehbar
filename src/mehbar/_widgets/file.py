import logging
import os
from pathlib import Path

import anyio

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import JSONInputMixin, WidgetBase


def _last_line(data: bytes) -> str | None:
    tail = data.rstrip()
    return tail[tail.rfind(b"\n") + 1 :].decode(errors="replace") or None


def read_last_line(
    path: Path, chunk_size: int = 4096, max_len: int = 64 * 1024
) -> str | None:
    """Returns the last non-empty line of the file, reading from its end.
    Only the last `max_len` bytes of longer lines are returned."""
    with open(path, "rb") as fhandle:
        try:
            end = fhandle.seek(0, os.SEEK_END)
        except OSError:  # e.g. procfs files that cannot seek to their end
            end = 0

        if end == 0:
            # procfs and sysfs files have no size, read them from the start
            fhandle.seek(0)
            return _last_line(fhandle.read(16 * max_len))

        buff = b""

        while end > 0:
            start = max(0, end - chunk_size)
            fhandle.seek(start)
            buff = fhandle.read(end - start) + buff
            end = start

            tail = buff.rstrip()

            # the first line may be incomplete unless the whole file is read
            if b"\n" in tail or end == 0:
                return _last_line(tail)

            if len(tail) > max_len:
                return tail[-max_len:].decode(errors="replace")

    return None


class WidgetFile(JSONInputMixin, WidgetBase):
    """Shows the last line of a file every interval. See `JSONInputMixin` for
    the format of the line."""

    UNIQUE = False
    MAX_FAILURES = 10
    TYPE = "file"
    POLLING = True

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        if not (path := self.cfg.get("path")):
            raise BarConfigError("'path' must be specified")

        self.path = Path(path)

    async def run(self):
        failed_cnt = 0

        while await self.sleep_interval():
            try:
                line = await anyio.to_thread.run_sync(read_last_line, self.path)
            except OSError as ex:
                # the file may be (re)created later
                logging.debug("cannot read from '%s': %s", self.path, ex)

                failed_cnt += 1
                if failed_cnt >= self.MAX_FAILURES:
                    raise
            else:
                failed_cnt = 0

                if line is not None:
                    self.set_content_from_line_i(line)
