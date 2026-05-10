import time

import anyio
from anyio.streams.text import TextReceiveStream

from mehbar.widget import JSONInputWidgetBase


class ExecWidgetBase(JSONInputWidgetBase):
    UNIQUE = False

    def __init__(
        self,
        interval: int,
        label_format: str,
        cmdline: str | list[str],
        ramp: list[str] | None = None,
        max_lps: int = 5,
    ):
        super().__init__(interval, label_format, ramp, max_lps)
        self.cmdline = cmdline


class WidgetExecTail(ExecWidgetBase):
    TYPE = "exec_tail"

    def __init__(
        self,
        label_format: str,
        cmdline: str | list[str],
        ramp: list[str] | None = None,
        max_lps: int = 5,
    ):
        super().__init__(0, label_format, cmdline, ramp, max_lps)

    async def run(self):
        async with await anyio.open_process(self.cmdline) as proc:
            lps = 0
            t0 = time.monotonic()

            if proc.stdout is not None:
                async for line in TextReceiveStream(proc.stdout):
                    if line.strip():
                        lps += 1
                        t1 = time.monotonic()

                        if (t1 - t0) >= 1:
                            lps = 0
                            t0 = t1

                        if lps <= self.max_lps:
                            await self.format_label_idle_json_async(line)


class WidgetExecRepeat(ExecWidgetBase):
    TYPE = "exec_repeat"

    async def run(self):
        while await self.sleep_interval():
            proc = await anyio.run_process(self.cmdline)
            if proc.stdout is not None:
                for line in proc.stdout.decode().splitlines()[: self.max_lps]:
                    if line.strip():
                        await self.format_label_idle_json_async(line)
                        break
