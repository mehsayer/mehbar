import logging
import time
from pathlib import Path

import anyio

from mehbar.widget import JSONInputWidgetBase


class WidgetFile(JSONInputWidgetBase):
    UNIQUE = False
    MAX_FAILURES = 10
    TYPE = "file"

    def __init__(
        self,
        interval: int,
        label_format: str,
        path: str | Path,
        ramp: list[str] | None = None,
        max_lps: int = 5,
    ):
        super().__init__(interval, label_format, ramp, max_lps)
        self.path = Path(path)

    async def run(self):

        failed_cnt = 0
        while await self.sleep_interval():
            # on first iteration, skip all lines up until the last one
            skip = self.path.is_file()
            try:
                async with await anyio.open_file(self.path) as fhandle:
                    lps = 0
                    t0 = time.monotonic()

                    line = ""
                    async for line_ in fhandle:
                        if line := line_.strip():
                            lps += 1
                            t1 = time.monotonic()

                            if (t1 - t0) >= 1:
                                lps = 0
                                t0 = t1

                            if not skip and lps <= self.max_lps:
                                await self.format_label_idle_json_async(line)
                    if skip:
                        skip = False
                        if line:
                            await self.format_label_idle_json_async(line)

            except Exception as ex:
                logging.debug("cannot read from '%s': %s", self.path, ex)

                failed_cnt += 1
                if failed_cnt >= self.MAX_FAILURES:
                    raise
            else:
                failed_cnt = 0
