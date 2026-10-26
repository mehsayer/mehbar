import logging
import os
import signal
import subprocess
from collections.abc import AsyncIterator

import anyio
from anyio.abc import ByteReceiveStream, Process

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import JSONInputMixin, WidgetBase


async def iter_lines(stream: ByteReceiveStream, max_len: int) -> AsyncIterator[str]:
    """Yields lines of text, lines longer than `max_len` bytes are dropped."""
    buff = b""
    # the rest of a line that is too long
    discarding = False

    async for chunk in stream:
        buff += chunk
        *lines, buff = buff.split(b"\n")

        for line in lines:
            if discarding:
                discarding = False
            else:
                yield line.decode(errors="replace")

        if len(buff) > max_len:
            buff = b""
            discarding = True

    if buff and not discarding:
        yield buff.decode(errors="replace")


async def open_command(cmdline: str | list[str]) -> Process:
    """Starts the command in its own process group, see `stop_command()`. A
    string is run by the shell."""
    return await anyio.open_process(
        cmdline,
        stdin=subprocess.DEVNULL,
        stderr=None,
        start_new_session=True,
    )


def _signal_group(proc: Process, signum: int):
    try:
        os.killpg(proc.pid, signum)
    except (ProcessLookupError, PermissionError):
        pass


async def stop_command(proc: Process, timeout: float = 1):
    """Terminates the command and processes it started, e.g. the rest of a
    shell pipeline, AnyIO would kill only the command. Waits for the command
    to exit: killing an exited, but not yet waited for, process races with
    the asyncio child watcher."""
    if proc.returncode is not None:
        return

    _signal_group(proc, signal.SIGTERM)

    with anyio.CancelScope(shield=True):
        with anyio.move_on_after(timeout):
            await proc.wait()

        if proc.returncode is None:
            _signal_group(proc, signal.SIGKILL)
            await proc.wait()


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
        self.cmdline = (
            cmdline if isinstance(cmdline, str) else [str(a) for a in cmdline]
        )

        max_lps = self.cfg.get("max_lps", self.MAX_LPS)

        if not isinstance(max_lps, (int, float)):
            raise BarConfigError("'max_lps' must be a number")
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

        async with await open_command(self.cmdline) as proc:
            try:
                async with anyio.create_task_group() as grp:
                    grp.start_soon(self._show_latest)

                    if proc.stdout is not None:
                        async for line in iter_lines(proc.stdout, self.MAX_LINE_LEN):
                            if line.strip():
                                self._latest_line = line
                                self._line_ready.set()

                    grp.cancel_scope.cancel()
            finally:
                await stop_command(proc)

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
    POLLING = True
    DEFAULT_TIMEOUT = 30
    MAX_LINE_LEN = 64 * 1024

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.timeout = self.cfg.get("timeout", self.DEFAULT_TIMEOUT)

        if not isinstance(self.timeout, (int, float)) or self.timeout <= 0:
            raise BarConfigError("'timeout' must be a positive number")

    async def _run_command(self) -> str | None:
        """Returns the first non-empty line of output. The rest is read, but
        not kept, so that the command does not block on a full pipe."""
        first_line = None

        async with await open_command(self.cmdline) as proc:
            try:
                if proc.stdout is not None:
                    async for line in iter_lines(proc.stdout, self.MAX_LINE_LEN):
                        if first_line is None and line.strip():
                            first_line = line

                await proc.wait()
            finally:
                await stop_command(proc)

        if proc.returncode:
            logging.debug(
                "widget '%s': command exited with status %d",
                self.widget_name,
                proc.returncode,
            )

        return first_line

    async def run(self):
        while await self.sleep_interval():
            with anyio.move_on_after(self.timeout) as scope:
                line = await self._run_command()

            if scope.cancelled_caught:
                logging.warning(
                    "widget '%s': command timed out after %s seconds",
                    self.widget_name,
                    self.timeout,
                )
                continue

            if line is not None:
                self.set_content_from_line_i(line)
