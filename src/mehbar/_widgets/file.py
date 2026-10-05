import logging
import os
from pathlib import Path

import anyio

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import JSONInputMixin, WidgetBase


def read_last_line(path: Path, chunk_size: int = 4096) -> str | None:
    """Returns the last non-empty line of the file, reading from its end."""
    with open(path, "rb") as fhandle:
        end = fhandle.seek(0, os.SEEK_END)
        buff = b""

        while end > 0:
            start = max(0, end - chunk_size)
            fhandle.seek(start)
            buff = fhandle.read(end - start) + buff
            end = start

            lines = [ln for ln in buff.splitlines() if ln.strip()]

            # the first line may be incomplete unless the whole file is read
            if len(lines) > 1 or (lines and end == 0):
                return lines[-1].decode(errors="replace")

    return None


class WidgetFile(JSONInputMixin, WidgetBase):
    """Shows the last line of a file every interval. See `JSONInputMixin` for
    the format of the line."""

    UNIQUE = False
    MAX_FAILURES = 10
    TYPE = "file"

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
