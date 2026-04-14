from mehbar.widgets import Widget


class WidgetExecTail(Widget):

    def __init__(self, label_format: str, cmdline: list[str], max_lps: int):
        super().__init__(0, label_format)
        self.max_lps = min(max_lps, 10)
        self.cmdline = cmdline
        self.proc = None

    async def run(self):

        try:
            self.proc = await asyncio.create_subprocess_exec(
                *self.cmdline,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )

            if self.proc.stdout is not None:

                lps = 0
                t0 = time.monotonic()
                while line := await self.proc.stdout.readline():
                    lps += 1
                    t1 = time.monotonic()

                    if (t1 - t0) >= 1:
                        lps = 0
                        t0 = t1

                    if lps <= self.max_lps:
                        try:
                            self.format_label_idle(**json.loads(line))
                            await asyncio.sleep(0.1)
                        except json.JSONDecodeError as ex:
                            logging.error("Failed to parse JSON: %s", str(ex))
            await self.proc.wait()
        finally:
            if self.proc and self.proc.returncode is None:
                self.proc.terminate()
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=3)
                except asyncio.TimeoutError:
                    self.proc.kill()

    def stop(self):
        if self.proc is not None and self.proc.returncode is None:
            self.proc.terminate()
        super().stop()


class WidgetExecRepeat(Widget):
    def __init__(
        self,
        interval: int,
        label_format: str,
        cmdline: list[str],
        max_lps: int,
    ):
        super().__init__(interval, label_format)
        self.max_lps = min(max_lps, 10)
        self.cmdline = cmdline
        self.proc = None


    async def update(self):
        self.proc = await asyncio.create_subprocess_exec(
            *self.cmdline,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        stdout, _ = await self.proc.communicate()

        for line in stdout.decode().splitlines()[: self.max_lps]:
            self.format_label_idle(**json.loads(line))
        self.proc = None

    def stop(self):
        if self.proc is not None and self.proc.returncode is None:
            self.proc.terminate()
        super().stop()
