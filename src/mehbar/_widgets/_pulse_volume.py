from pulsectl_asyncio import PulseAsync
from mehbar.widgets import BarWidget
import asyncio

class BarWidgetPulseVolume(BarWidget):

    def __init__(self,
                 sink_name: int,
                 max_vol: int,
                 vol_delta: int,
                 label_format: str,
                 ramp: list[str] | None = None):
        super().__init__(0, label_format, ramp)
        self.sink_name = sink_name
        self.max_vol = max(min(max_vol, 200), 20)

        self.aio_loop = None

        self.vol_delta = vol_delta / 100

        self.sink_action_queue = asyncio.Queue(maxsize=8)

        self.results: list[list[str]] = [[], []]

        for vol in range(self.max_vol + 1):
            muted = ""
            unmuted = ""

            if ramp is not None and (nramp := len(ramp) - 1) > 0:
                ramp_idx = int(min(vol, max_vol - 1) / (max_vol / nramp))
                unmuted = ramp[1:][ramp_idx]
                muted = ramp[0]

            self.results[0].append(self.vformat_label(percent=vol,
                                                      ramp=unmuted))
            self.results[1].append(self.vformat_label(percent=vol,
                                                      ramp=muted))

        self.onclick_call(3, self.action_mute)
        self.onscroll_call(self.action_volume_down, self.action_volume_up)

        # FIXME: replace all this with functools.partialmethod


    def action_volume_up(self):
        if self.aio_loop is not None:
            self.aio_loop.call_soon_threadsafe(self._sink_action,
                                               SinkAction.VOLUME,
                                               self.vol_delta)

    def action_volume_down(self):
        if self.aio_loop is not None:
            self.aio_loop.call_soon_threadsafe(self._sink_action,
                                               SinkAction.VOLUME,
                                               -self.vol_delta)

    def action_mute(self, *args):
        if self.aio_loop is not None:
            self.aio_loop.call_soon_threadsafe(self._sink_action,
                                               SinkAction.MUTE, 0)

    def _sink_action(self, action: SinkAction, value: int):
        if not self.sink_action_queue.full():
            self.sink_action_queue.put_nowait((action, value))


    async def _listen(self, handle: PulseAsync):

        sink_idx = (await handle.get_sink_by_name(self.sink_name)).index

        async def _update_volume_label(handle: PulseAsync):

            sink = await handle.sink_info(sink_idx)

            vol = round(sink.volume.value_flat * 100)

            if vol <= self.max_vol:
                self.set_label_idle(self.results[sink.mute][vol])
                await asyncio.sleep(0.15)

        # if we start with volume level, that's more than 1.0 (100)
        self.set_label_idle(self.results[0][100])
        await _update_volume_label(handle)

        async for event in handle.subscribe_events("sink"):
            if event.index == sink_idx and event.t == "change":
                await _update_volume_label(handle)


    async def _consume(self, handle: PulseAsync):

        sink = await handle.get_sink_by_name(self.sink_name)

        while True:
            action, delta = await self.sink_action_queue.get()

            if action == SinkAction.VOLUME:
                vol = round(max(0, sink.volume.value_flat + delta) * 100)

                if vol >= 0 and vol <= self.max_vol:
                    await handle.volume_change_all_chans(sink, delta)
                    await asyncio.sleep(0.1)
            elif action == SinkAction.MUTE:
                await handle.mute(sink, sink.mute == 0)


    async def run(self):
        self.aio_loop = asyncio.get_running_loop()

        async with PulseAsync("poll-volume") as pulse:
            async with asyncio.TaskGroup() as grp:
                tlisten = grp.create_task(self._listen(pulse))
                tconsume = grp.create_task(self._consume(pulse))
