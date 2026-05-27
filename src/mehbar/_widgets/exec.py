import time

import anyio
from anyio.streams.text import TextReceiveStream

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import JSONInputMixin, WidgetBase


class ExecWidgetBase(JSONInputMixin, WidgetBase):
    UNIQUE = False
    MAX_LPS = 10

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        if (cmdline := self.cfg.get("cmdline")) is None or not cmdline:
            raise BarConfigError("'cmdline' must be specified")
        self.cmdline = cmdline

        max_lps = self.cfg.get("max_lps", self.MAX_LPS)
        self.max_lps = max(min(max_lps, self.MAX_LPS), 1)


class WidgetExecTail(ExecWidgetBase):
    TYPE = "exec_tail"

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
                            await self.set_content_json_i(line)


class WidgetExecRepeat(ExecWidgetBase):
    TYPE = "exec_repeat"

    async def run(self):
        while await self.sleep_interval():
            proc = await anyio.run_process(self.cmdline)
            if proc.stdout is not None:
                for line in proc.stdout.decode().splitlines()[: self.max_lps]:
                    if line.strip():
                        await self.set_content_json_i(line)
                        break
