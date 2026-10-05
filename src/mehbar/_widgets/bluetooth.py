import anyio
from anyio.abc import ObjectSendStream

from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase

from ._bluetooth import BluetoothInfo, BluezBackend


class WidgetBluetoothStatus(WidgetBase):
    """Shows Bluetooth status, updated on BlueZ D-Bus signals. Ramp entries
    are for the status: off, on, connected. If the interval is set, the
    status is polled as well."""

    TYPE = "bluetooth-status"

    # coalesce bursts of signals, e.g. when a device connects
    DEBOUNCE = 0.2

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.backend = BluezBackend()

    def ramp_index(self, ramp_level: int) -> int | None:
        if ramp_level < 0 or not self.ramp:
            return None
        return min(ramp_level, len(self.ramp) - 1)

    def _show(self, info: BluetoothInfo):
        self.set_new_content_i(
            int(info.status),
            status=info.status.name.lower(),
            name=info.name,
            alias=info.alias,
            battery=info.bat_percent if info.bat_percent >= 0 else None,
            volume=info.volume if info.volume >= 0 else None,
        )

    async def _poll(self, sstream: ObjectSendStream):
        # the first iteration does not sleep, the status is queried anyway
        await self.sleep_interval()

        while await self.sleep_interval():
            try:
                sstream.send_nowait(None)
            except anyio.WouldBlock:
                pass

    async def run(self):
        sstream, rstream = anyio.create_memory_object_stream[None](1)

        def _notify():
            try:
                sstream.send_nowait(None)
            except (anyio.WouldBlock, anyio.ClosedResourceError):
                pass

        # connect to the bus in a worker thread, it blocks
        await anyio.to_thread.run_sync(lambda: self.backend.bus)

        sub_ids = self.backend.subscribe(lambda: self.call_soon(_notify))

        try:
            async with sstream, rstream, anyio.create_task_group() as grp:
                grp.start_soon(self._poll, sstream)

                self._show(await anyio.to_thread.run_sync(self.backend.get_info))

                async for _ in rstream:
                    await anyio.sleep(self.DEBOUNCE)
                    self._show(await anyio.to_thread.run_sync(self.backend.get_info))
        finally:
            self.backend.unsubscribe(sub_ids)
