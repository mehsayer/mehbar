import asyncio
from functools import partial

import anyio
from anyio.abc import ObjectReceiveStream, ObjectSendStream

from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


class WidgetPulseVolume(WidgetBase):
    """Shows the sink volume. Scroll changes the volume, right click toggles
    mute. The first ramp entry is shown when muted, the rest is spread over
    0..`max_volume` percent."""

    TYPE = "pulse_volume"
    DEFAULT_VOLUME = 100
    MAX_VOLUME = 200
    MIN_VOLUME = 20
    DEFAULT_DELTA = 10
    QUEUE_SIZE = 8
    CONNECT_TIMEOUT = 5

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.sink_name = self.cfg.get("sink_name", "@DEFAULT_SINK@")

        max_volume_ = self.cfg.get("max_volume", self.DEFAULT_VOLUME)

        self.max_volume = max(min(max_volume_, self.MAX_VOLUME), self.MIN_VOLUME)

        self.vol_delta = self.cfg.get("volume_delta", self.DEFAULT_DELTA) / 100

        self._sstream: ObjectSendStream[float | None] | None = None

        self.onclick_call(3, self.call_soon, self._request, None)
        self.onscroll_call(
            partial(self.call_soon, self._request, self.vol_delta),
            partial(self.call_soon, self._request, -self.vol_delta),
        )

    def ramp_index(self, ramp_level: int) -> int | None:
        if ramp_level < 0 or not self.ramp:
            return None

        if ramp_level > self.max_volume or len(self.ramp) == 1:
            return 0

        level = min(ramp_level, self.max_volume - 1)
        return int(level / (self.max_volume / (len(self.ramp) - 1))) + 1

    def _request(self, delta: float | None):
        """Requests a volume change by `delta`, `None` toggles mute."""
        if self._sstream is not None:
            try:
                self._sstream.send_nowait(delta)
            except (anyio.WouldBlock, anyio.ClosedResourceError):
                pass

    @staticmethod
    async def _op(coro_func, *args):
        """Runs a pulse operation. pulsectl-asyncio leaves a dangling C
        callback behind if an operation is cancelled midway, which crashes
        the process later, so operations are shielded from cancellation."""
        with anyio.CancelScope(shield=True):
            return await coro_func(*args)

    async def _show_volume(self, pulse):
        sink = await self._op(pulse.sink_info, self._sink_idx)

        volume = round(sink.volume.value_flat * 100)

        if sink.mute:
            ramp_level = self.max_volume + 1
        else:
            ramp_level = min(volume, self.max_volume)

        self.set_new_content_i(ramp_level, percent=volume, muted=bool(sink.mute))

    async def _resolve_sink(self, pulse):
        self._sink_idx = (await self._op(pulse.get_sink_by_name, self.sink_name)).index

    async def _pump_events(self, pulse, wake: ObjectSendStream[None]):
        """Runs as a plain asyncio task, out of reach of AnyIO cancellation:
        cancelling `subscribe_events()` while connected crashes the process,
        see `_op()`. Closes `wake` when the subscription ends."""
        with wake:
            async for event in pulse.subscribe_events("sink", "server"):
                if event.facility == "server" or (
                    event.index == self._sink_idx and event.t == "remove"
                ):
                    # the default sink may have changed
                    self._resolve_needed = True
                elif event.index == self._sink_idx and event.t == "change":
                    self._update_needed = True
                else:
                    continue

                try:
                    wake.send_nowait(None)
                except anyio.WouldBlock:
                    pass

    async def _listen(self, pulse, wake: ObjectReceiveStream[None]):
        await self._resolve_sink(pulse)
        await self._show_volume(pulse)

        async with wake:
            async for _ in wake:
                if self._resolve_needed:
                    self._resolve_needed = False
                    await self._resolve_sink(pulse)
                    self._update_needed = True

                if self._update_needed:
                    self._update_needed = False
                    await self._show_volume(pulse)

        raise ConnectionError("disconnected from the sound server")

    async def _consume(self, pulse, rstream: ObjectReceiveStream[float | None]):
        async with rstream:
            async for delta in rstream:
                # sink state changes outside of the widget too
                sink = await self._op(pulse.get_sink_by_name, self.sink_name)

                if delta is None:
                    await self._op(pulse.mute, sink, not sink.mute)
                else:
                    current = sink.volume.value_flat
                    target = max(0.0, min(current + delta, self.max_volume / 100))

                    if abs(target - current) >= 0.005:
                        await self._op(
                            pulse.volume_change_all_chans, sink, target - current
                        )

    async def run(self):
        from pulsectl_asyncio import PulseAsync

        self._sink_idx = -1
        self._resolve_needed = False
        self._update_needed = False

        self._sstream, rstream = anyio.create_memory_object_stream[float | None](
            self.QUEUE_SIZE
        )
        wake_send, wake_recv = anyio.create_memory_object_stream[None](1)

        pulse = PulseAsync("mehbar-volume")
        pump = None

        try:
            with anyio.CancelScope(shield=True):
                await pulse.connect(timeout=self.CONNECT_TIMEOUT)

            pump = asyncio.ensure_future(self._pump_events(pulse, wake_send))

            async with anyio.create_task_group() as grp:
                grp.start_soon(self._consume, pulse, rstream)
                await self._listen(pulse, wake_recv)
        finally:
            self._sstream.close()
            self._sstream = None

            # close the connection before the subscription is cancelled, so
            # that it does not try to unsubscribe
            pulse.close()

            if pump is not None:
                pump.cancel()
                # the outcome does not matter at this point
                pump.add_done_callback(lambda t: t.cancelled() or t.exception())
