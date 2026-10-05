import logging
import subprocess
from collections.abc import AsyncIterator

import anyio
from anyio.abc import ByteReceiveStream

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import JSONInputMixin, WidgetBase


async def iter_lines(stream: ByteReceiveStream, max_len: int) -> AsyncIterator[str]:
    """Yields lines of text, lines longer than `max_len` bytes are dropped."""
    buff = b""

    async for chunk in stream:
        buff += chunk
        *lines, buff = buff.split(b"\n")

        for line in lines:
            yield line.decode(errors="replace")

        if len(buff) > max_len:
            buff = b""

    if buff:
        yield buff.decode(errors="replace")


class ExecWidgetBase(JSONInputMixin, WidgetBase):
    """Runs a command, shows lines of its output. See `JSONInputMixin` for
    the format of the lines."""

    UNIQUE = False
    MAX_LPS = 10

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        cmdline = self.cfg.get("cmdline")

        if not cmdline or not isinstance(cmdline, (str, list)):
            raise BarConfigError("'cmdline' must be specified")

        # a string is run by the shell
        self.cmdline = cmdline if isinstance(cmdline, str) else [str(a) for a in cmdline]

        max_lps = self.cfg.get("max_lps", self.MAX_LPS)
        self.max_lps = max(min(max_lps, self.MAX_LPS), 1)


class WidgetExecTail(ExecWidgetBase):
    """Shows the most recent line of output of a long running command, at most
    `max_lps` lines per second."""

    TYPE = "exec_tail"
    MAX_LINE_LEN = 64 * 1024

    async def _show_latest(self):
        while True:
            await self._line_ready.wait()
            self._line_ready = anyio.Event()

            line, self._latest_line = self._latest_line, None
            if line is not None:
                self.set_content_from_line_i(line)

            await anyio.sleep(1 / self.max_lps)

    async def run(self):
        self._latest_line: str | None = None
        self._line_ready = anyio.Event()

        async with await anyio.open_process(
            self.cmdline, stdin=subprocess.DEVNULL, stderr=None
        ) as proc:
            async with anyio.create_task_group() as grp:
                grp.start_soon(self._show_latest)

                if proc.stdout is not None:
                    async for line in iter_lines(proc.stdout, self.MAX_LINE_LEN):
                        if line.strip():
                            self._latest_line = line
                            self._line_ready.set()

                grp.cancel_scope.cancel()

        if self._latest_line is not None:
            self.set_content_from_line_i(self._latest_line)

        if proc.returncode:
            logging.warning(
                "widget '%s': command exited with status %d",
                self.widget_name,
                proc.returncode,
            )


class WidgetExecRepeat(ExecWidgetBase):
    """Runs the command every interval, shows the first line of its output."""

    TYPE = "exec_repeat"
    DEFAULT_TIMEOUT = 30

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.timeout = self.cfg.get("timeout", self.DEFAULT_TIMEOUT)

    async def run(self):
        while await self.sleep_interval():
            result = None

            with anyio.move_on_after(self.timeout):
                result = await anyio.run_process(
                    self.cmdline, stdin=subprocess.DEVNULL, stderr=None, check=False
                )

            if result is None:
                logging.warning(
                    "widget '%s': command timed out after %s seconds",
                    self.widget_name,
                    self.timeout,
                )
                continue

            if result.returncode:
                logging.debug(
                    "widget '%s': command exited with status %d",
                    self.widget_name,
                    result.returncode,
                )

            for line in result.stdout.decode(errors="replace").splitlines():
                if line.strip():
                    self.set_content_from_line_i(line)
                    break
