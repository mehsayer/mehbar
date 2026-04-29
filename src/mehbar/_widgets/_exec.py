import json
import logging
import time

import anyio
from anyio.streams.text import TextReceiveStream

from mehbar.widgets import Widget


class WidgetExecBase(Widget):
    def __init__(
        self,
        interval: int,
        label_format: str,
        cmdline: str | list[str],
        max_lps: int,
    ):
        super().__init__(interval, label_format)
        self.max_lps = min(max_lps, 10)
        self.cmdline = cmdline

    async def format_label_idle_json_async(self, json_str: str):
        try:
            self.format_label_idle(**json.loads(json_str))
        except json.JSONDecodeError as ex:
            logging.error("failed to parse JSON input: %s", ex)
        finally:
            await anyio.sleep(0.1)


class WidgetExecTail(WidgetExecBase):
    def __init__(self, label_format: str, cmdline: list[str], max_lps: int):
        super().__init__(0, label_format, cmdline, max_lps)

    async def run(self):
        async with await anyio.open_process(self.cmdline) as proc:
            lps = 0
            t0 = time.monotonic()

            if proc.stdout is not None:
                async for line in TextReceiveStream(proc.stdout):
                    lps += 1
                    t1 = time.monotonic()

                    if (t1 - t0) >= 1:
                        lps = 0
                        t0 = t1

                    if line and lps <= self.max_lps:
                        await self.format_label_idle_json_async(line)


class WidgetExecRepeat(WidgetExecBase):
    async def run(self):
        while await self.sleep_interval():
            proc = await anyio.run_process(self.cmdline)
            if proc.stdout is not None:
                for line in proc.stdout.decode().splitlines()[: self.max_lps]:
                    if line.strip():
                        await self.format_label_idle_json_async(line)
                        break
