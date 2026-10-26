import anyio
from psutil import disk_usage

from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


class WidgetDiskUsage(WidgetBase):
    """Shows disk usage of the file system at `path`. The widget is hidden
    while the file system is not available, e.g. a network share does not
    respond."""

    TYPE = "disk_usage"
    DEFAULT_TOOLTIP = "{path}: {used_gib} of {total_gib} GiB used, {avail_gib} GiB free"
    POLLING = True
    TIMEOUT = 5

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)
        self.path = self.cfg.get("path", "/")
        # a read of a hung network file system blocks its thread until the
        # file system responds, AnyIO only stops waiting for it
        self._reading = False

    def _read(self):
        try:
            return disk_usage(self.path)
        finally:
            self._reading = False

    async def _disk_usage(self):
        # no more threads while the previous read is stuck
        if not self._reading:
            self._reading = True

            with anyio.move_on_after(self.TIMEOUT):
                return await anyio.to_thread.run_sync(
                    self._read, abandon_on_cancel=True
                )

        raise TimeoutError(f"'{self.path}' does not respond")

    async def run(self):
        seen = False

        while await self.sleep_interval():
            try:
                dusage = await self._disk_usage()
            except (OSError, TimeoutError):
                # a wrong path disables the widget
                if not seen:
                    raise
                self.set_available_i(False)
                continue

            seen = True
            self.set_available_i(True)

            percent = min(round(dusage.percent), 100)

            self.set_new_content_i(
                percent,
                used_gib=round(dusage.used / (1024**3), 1),
                total_gib=round(dusage.total / (1024**3), 1),
                avail_gib=round(dusage.free / (1024**3), 1),
                percent=percent,
                path=self.path,
            )
