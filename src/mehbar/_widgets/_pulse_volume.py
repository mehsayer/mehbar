from functools import partial

import anyio
from pulsectl_asyncio import PulseAsync

from mehbar.widgets import Widget


class WidgetPulseVolume(Widget):
    CMD_BASE = 128

    def __init__(
        self,
        sink_name: int,
        max_vol: int,
        vol_delta: int,
        label_format: str,
        ramp: list[str] | None = None,
    ):
        super().__init__(0, label_format, ramp)
        self.sink_name = sink_name
        self.max_vol = max(min(max_vol, 200), 20)

        self.vol_delta = vol_delta / 100

        self.sstream, self.rstream = anyio.create_memory_object_stream[int](8)

        self.results: list[list[str]] = [[], []]

        for vol in range(self.max_vol + 1):
            muted = ""
            unmuted = ""

            if ramp is not None and (nramp := len(ramp) - 1) > 0:
                ramp_idx = int(min(vol, max_vol - 1) / (max_vol / nramp))
                unmuted = ramp[1:][ramp_idx]
                muted = ramp[0]

            self.results[0].append(self.vsformat(percent=vol, ramp=unmuted))
            self.results[1].append(self.vsformat(percent=vol, ramp=muted))

        self.onclick_call(
            3, partial(self.elt_run_sync, self._sink_action, self.CMD_BASE)
        )

        self.onscroll_call(
            partial(
                self.elt_run_sync, self._sink_action, self.CMD_BASE - self.vol_delta
            ),
            partial(
                self.elt_run_sync, self._sink_action, self.CMD_BASE + self.vol_delta
            ),
        )

    def _sink_action(self, cmd: int):
        try:
            self.sstream.send_nowait(cmd)
        except anyio.WouldBlock:
            pass

    async def _listen(self, handle: PulseAsync):

        sink_idx = (await handle.get_sink_by_name(self.sink_name)).index

        async def _update_volume_label(handle: PulseAsync):

            sink = await handle.sink_info(sink_idx)

            vol = round(sink.volume.value_flat * 100)

            if vol <= self.max_vol:
                self.set_label_idle(self.results[sink.mute][vol])
                await anyio.sleep(0.1)

        # if we start with volume level, that's more than 1.0 (100)
        self.set_label_idle(self.results[0][100])

        await _update_volume_label(handle)

        async for event in handle.subscribe_events("sink"):
            if event.index == sink_idx and event.t == "change":
                await _update_volume_label(handle)

    async def _consume(self, handle: PulseAsync):

        sink = await handle.get_sink_by_name(self.sink_name)

        async with self.rstream:
            async for cmd in self.rstream:
                if cmd == self.CMD_BASE:
                    await handle.mute(sink, sink.mute == 0)
                else:
                    delta = cmd - self.CMD_BASE
                    vol = round(max(0, sink.volume.value_flat + delta) * 100)

                    if vol >= 0 and vol <= self.max_vol:
                        await handle.volume_change_all_chans(sink, delta)
                        await anyio.sleep(0.1)

    async def run(self):
        async with PulseAsync("poll-volume") as pulse:
            async with anyio.create_task_group() as grp:
                grp.start_soon(self._listen, pulse)
                grp.start_soon(self._consume, pulse)
