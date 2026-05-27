from functools import partial

import anyio

from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase, WidgetContent


class WidgetPulseVolume(WidgetBase):
    TYPE = "pulse_volume"
    CMD_BASE = 128
    DEFAULT_VOLUME = 100
    MAX_VOLUME = 200
    MIN_VOLUME = 20
    DEFAULT_DELTA = 10

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.sink_name = self.cfg.get("sink_name", "@DEFAULT_SINK@")

        max_volume_ = self.cfg.get("max_volume", self.DEFAULT_VOLUME)

        self.max_volume = max(min(max_volume_, self.MAX_VOLUME), self.MIN_VOLUME)

        self.vol_delta = self.cfg.get("volume_delta", self.DEFAULT_DELTA) / 100

        self.sstream, self.rstream = anyio.create_memory_object_stream[int](8)

        volume_action = partial(self.elt_run_sync, self._sink_action)

        volume_down = partial(volume_action, self.CMD_BASE - self.vol_delta)
        volume_up = partial(volume_action, self.CMD_BASE + self.vol_delta)
        volume_mute = partial(volume_action, self.CMD_BASE)

        self.onclick_call(3, volume_mute)
        self.onscroll_call(volume_down, volume_up)

    def get_ramp(self, ramp_level: int = -1) -> WidgetContent | None:

        if ramp_level not in self.ramp_index_cache:
            ramp = self.cfg.get("ramp")
            content = None

            if ramp is not None and ramp:
                if ramp_level >= 0 and ramp is not None and ramp:
                    if ramp_level <= self.max_volume:
                        level_ = min(ramp_level, self.max_volume - 1)
                        idx = int(level_ / (self.max_volume / (len(ramp) - 1))) + 1
                    else:
                        idx = 0
                    content = WidgetContent.parse(ramp[idx])

            self.ramp_index_cache[ramp_level] = content

        return self.ramp_index_cache[ramp_level]

    def _sink_action(self, cmd: int):

        try:
            self.sstream.send_nowait(cmd)
        except anyio.WouldBlock:
            pass

    async def _listen(self, handle):

        sink_idx = (await handle.get_sink_by_name(self.sink_name)).index

        async def _update_volume_label(handle):

            sink = await handle.sink_info(sink_idx)

            volume = round(sink.volume.value_flat * 100)

            if volume <= self.max_volume:
                if sink.mute == 0:
                    ramp_level = volume
                else:
                    ramp_level = self.max_volume + 1
                self.set_new_content_i(percent=volume, ramp_level=ramp_level)
                await anyio.sleep(0.1)

        self.set_new_content_i(percent=self.max_volume, ramp_level=self.max_volume)

        await _update_volume_label(handle)

        async for event in handle.subscribe_events("sink"):
            if event.index == sink_idx and event.t == "change":
                await _update_volume_label(handle)

    async def _consume(self, handle):

        sink = await handle.get_sink_by_name(self.sink_name)

        async with self.rstream:
            async for cmd in self.rstream:
                if cmd == self.CMD_BASE:
                    await handle.mute(sink, sink.mute == 0)
                else:
                    delta = cmd - self.CMD_BASE
                    vol = round(max(0, sink.volume.value_flat + delta) * 100)

                    if vol >= 0 and vol <= self.max_volume:
                        await handle.volume_change_all_chans(sink, delta)
                        await anyio.sleep(0.1)

    async def run(self):

        from pulsectl_asyncio import PulseAsync

        async with PulseAsync("poll-volume") as pulse:
            async with anyio.create_task_group() as grp:
                grp.start_soon(self._listen, pulse)
                grp.start_soon(self._consume, pulse)
