#!/usr/bin/env python3
from __future__ import annotations
from typing import Sequence, Any, Mapping, Callable
from datetime import datetime
from enum import Enum
from itertools import compress
from pathlib import Path
from ctypes import CDLL
from inspect import Parameter, signature

import asyncio
import json
import re
import string
import sys
import threading
import time
import logging
import signal
import os

LOG_LEVEL = os.getenv("MEHBAR_LOG_LEVEL", "DEBUG")

logging.basicConfig(
    format="[%(asctime)s] *%(levelname)s*: %(message)s", level=getattr(logging, LOG_LEVEL, logging.INFO)
)

try:
    from i3ipc import (
        Event,
        InputEvent,
        ModeEvent,
        WindowEvent,
        WorkspaceEvent,
        Con,
    )
    from i3ipc.aio import Connection
except ImportError:
    pass

try:
    from pulsectl_asyncio import PulseAsync
except ImportError:
    pass

try:
    import psutil
except ImportError:
    pass


import gi
# GSK_RENDERER=cairo GDK_BACKEND=wayland
# logging.critical("DS")
cdll_failed = set()

for soname in ["libgtk4-layer-shell.so.0", "libgtk4-layer-shell.so.0"]:
    try:
        CDLL(soname)
        break
    except OSError as ex:

        cdll_failed.add(soname)
    else:
        cdll_failed.clear()

if cdll_failed:
    logging.critical("failed to load GTK4 layer shell library, tried: %s", ", ".join(cdll_failed))

    sys.exit(1)
try:
    gi.require_version("Gtk", "4.0")
    gi.require_version("Gtk4LayerShell", "1.0")
    from gi.repository import Gtk, Gdk, Gio, GLib
    from gi.repository import Gtk4LayerShell
except ValueError as ex:
    logging.critical(ex)
    sys.exit(1)

try:
    gi.require_version("Playerctl", "2.0")
    from gi.repository import Playerctl
except ImportError:
    pass

def overlay_dict_r(bottom: dict[Any, Any], top: dict[Any, Any], max_depth: int = 10, depth: int = 0):
    if depth > max_depth:
        raise ValueError(f"maximum nesting depth exceeded: {max_depth}")

    for ktop, vtop in top.items():
        if isinstance(vtop, dict):
            if ktop not in bottom or not isinstance(bottom[ktop], dict):
                bottom[ktop] = {}
            overlay_dict_r(bottom[ktop], vtop, max_depth, depth + 1)
        else:
            bottom[ktop] = vtop


# class HashDict(dict):
#     def __hash__(self):
#         return hash(frozenset(self))

class BarConfigError(Exception):
    pass

class SinkAction(Enum):
    VOLUME = 1
    MUTE = 2

class GestureMouseClick(Gtk.GestureClick, Gtk.GestureSingle):
    pass

class BarWindgetInterface:

    async def run(self):
        pass

    def update(self):
        pass

    def vformat_label(self, **kwargs):
        pass

    def vupdate_label(self, **kwargs):
        pass

    def stop(self):
        pass


class Action:
    def run(self):
        pass

class CallableAction(Action):

    def __init__(self, func: Callable, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.func(*self.args, **self.kwargs)


class BarWidget(BarWindgetInterface, Gtk.Label):

    def __init__(
        self,
        interval: int = 0,
        label_format: str | None = None,
        ramp: list[str] | None = None,
    ):
        super().__init__()

        self._run = True
        self._last_value: Any | None = None
        self.cache: dict[str, Any] = {}
        self.formatter = OptionalFormatter()
        self.interval = max(int(interval), 0)
        self.label_format = label_format if label_format is not None else ""
        self.ramp = ramp

        self.set_xalign(0.5)
        self.set_yalign(0.5)
        self.set_single_line_mode(True)

        self.add_css_class("bar-widget")

        # set_ellipsize PANGO_ELLIPSIZE_END

    async def run(self):
        self.update()

        if self.interval > 0:
            while self._run:
                self.update()
                await asyncio.sleep(self.interval)

    def update(self):
        pass

    def set_label_idle(self, label: str) -> bool:
        """Calls Widget.set_label and returns False, so that it can be removed
        from event sources.
        """
        super().set_label(label)
        return GLib.SOURCE_REMOVE

    def vformat_label(self, **kwargs):
        return self.formatter.format(self.label_format, **kwargs)

    def vupdate_label(self, **kwargs):
        GLib.idle_add(self.set_label_idle, self.vformat_label(**kwargs))

    def stop(self):
        self._run = False

    def onclick(self, button: int, action: Action):
        controller = GestureMouseClick()
        controller.set_button(button)
        controller.connect("pressed", lambda *args: action.run())
        self.add_controller(controller)

    def onscroll(self, action_up: Action, action_down: Action):

        def _scroll(controller: Gtk.EventControllerScroll, dx: float, dy: float):
            if dy > 0:
                action_up.run()
            else:
                action_down.run()

        controller = Gtk.EventControllerScroll.new(Gtk.EventControllerScrollFlags.VERTICAL)
        controller.connect("scroll", _scroll)
        self.add_controller(controller)


class RewriteMixin:
    def __init__(self, *args, rewrite: dict[str, str], **kwargs):
        super().__init__(*args, **kwargs)
        self._rewrite = rewrite

    def rewrite(self, text: str) -> str:
        result = text
        if text is not None:
            for pattern, repl in self._rewrite.items():
                if re.match(pattern, text) is not None:
                    result = re.sub(pattern, repl, text)
                    break
        return result


class I3ListenerMixin:
    def __init__(self, *args, i3_conn: Connection, **kwargs):
        super().__init__(*args, **kwargs)
        self._i3_conn = i3_conn

    async def get_i3_conn(self):
        if self._i3_conn is None:
            self._i3_conn = await Connection().connect()
        return self._i3_conn


class OptionalFormatter(string.Formatter):
    """Like the default stripng formatter that you know and love but silently
    skips missing fields.
    """

    def vformat(
        self,
        format_string: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> str:

        if args:
            raise ValueError("non-keyword arguments are not supported")

        unparsed = str()

        for literal, fld, _, _ in self.parse(format_string):
            if not kwargs or fld is None or fld not in kwargs:
                unparsed += literal
            elif fld in kwargs and str(kwargs[fld]):
                unparsed += literal + "{" + fld + "}"

        return super().vformat(unparsed, args, kwargs)


class BarWidgetStatic(BarWidget):
    def __init__(self, label_format: str):
        super().__init__(0, label_format)
        self.set_label(self.label_format)



class BarWidgetTemperature(BarWidget):
    def __init__(
        self,
        zone: int,
        max_temp: int,
        interval: int,
        label_format: str,
        ramp: list[str] | None = None,
    ):
        super().__init__(interval, label_format, ramp)

        self.path_term = Path(f"/sys/class/thermal/thermal_zone{zone}/temp")

        self.max_temp = max(min(max_temp, 200), 0)

        self.results = []

        # pre-generate formatted strings for all possible temperatures.
        # this uses O(max_temp) memory for O(1) update speed.
        # ranges of possible values in widgets are raltively narrow and do not
        # exceed at most a copule of hundreds of short strings, hence the
        # memory overhead is negligible.
        for temp in range(max_temp + 1):
            ramp_val = str()

            if ramp is not None and (nramp := len(ramp)) > 0:
                ramp_idx = int(min(temp, max_temp - 1) / (max_temp / nramp))
                ramp_val = ramp[ramp_idx]

            self.results.append(
                self.vformat_label(celsius=temp, ramp=ramp_val)
            )

    def update(self):
        temp = 0
        with open(self.path_term, "r", encoding="ascii") as fhandle:
            temp = int(fhandle.readline()) // 1000

        if self._last_value != temp:
            self._last_value = temp
            GLib.idle_add(self.set_label_idle,
                          self.results[min(temp, self.max_temp)])


class BarWidgetCPUPercentage(BarWidget):
    def update(self):
        percentage = round(psutil.cpu_percent())

        if self._last_value != percentage:
            self._last_value = percentage
            self.vupdate_label(percent=percentage)


class BarWidgetMemoryUsage(BarWidget):
    def update(self):
        vmem = psutil.virtual_memory()

        if self._last_value != vmem.used:
            self._last_value = vmem.used

            used_mib = vmem.used / (1024**2)
            total_mib = vmem.total / (1024**2)
            avail_mib = vmem.available / (1024**2)

            self.vupdate_label(
                used_mib=round(used_mib),
                used_gib=round(used_mib / 1024, 1),
                total_mib=round(total_mib),
                total_gib=round(total_mib / 1024, 1),
                avail_mib=round(avail_mib),
                avail_gib=round(avail_mib / 1024, 1),
                percent=round(vmem.percent),
            )

class BarWidgetDiskUsage(BarWidget):

    def __init__(self, interval: int, label_format: str, path: str):
        super().__init__(interval, label_format)
        self.path = path

    def update(self):
        dusage = psutil.disk_usage(self.path)

        if self._last_value != dusage.used:
            self._last_value = dusage.used

            self.vupdate_label(
                used_gib=round(dusage.used / (1024**3), 1),
                total_gib=round(dusage.total / (1024**3), 1),
                avail_gib=round(dusage.free / (1024**3), 1),
                percent=round(dusage.percent),
            )


class BarWidgetWifiSignal(BarWidget):

    MAX_SIGNAL = 100

    def __init__(
        self,
        interval: int,
        iface: str,
        label_format: str,
        ramp: list[str] | None = None,
    ):
        super().__init__(interval, label_format, ramp)
        self.iface = iface

        self.results = []

        if ramp is not None and (nramp := len(ramp) - 1) > 0:
            self.results.append(self.vformat_label(signal=0, ramp=ramp[0]))

            for sig in range(1, self.MAX_SIGNAL + 1):

                if ramp is not None and (nramp := len(ramp) - 1) > 0:
                    ramp_idx = int(min(sig, self.MAX_SIGNAL - 1) / (self.MAX_SIGNAL / nramp))
                    ramp_val = ramp[1:][ramp_idx]

                self.results.append(
                    self.vformat_label(signal=sig, ramp=ramp_val)
                )

    def update(self):
        sig = 0

        with open("/proc/net/wireless", "r", encoding="ascii") as fhandle:
            for ln in fhandle:
                ln_split = ln.split(maxsplit=3)
                if len(ln_split) > 3:
                    _iface, _, _sig, _ = ln_split

                    if _iface[:-1] == self.iface:
                        sig = int(_sig[:-1])
                        break

        if self._last_value != sig:
            self._last_value = sig
            GLib.idle_add(self.set_label_idle, self.results[sig])


class BarWidgetDateTime(BarWidget):
    def __init__(self, interval: int, label_format: str, datetime_format: str):
        super().__init__(interval, label_format)
        self.datetime_format = datetime_format

    def update(self):
        datetime_str = datetime.now().strftime(self.datetime_format)

        if self._last_value != datetime_str:
            self._last_value = datetime_str
            self.vupdate_label(datetime=datetime_str)

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

        self.onclick(3, CallableAction(self.action_mute))
        self.onscroll(CallableAction(self.action_volume_down),
                      CallableAction(self.action_volume_up))


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


    async def _listen_sink_events(self, handle: PulseAsync):

        sink_idx = (await handle.get_sink_by_name(self.sink_name)).index

        async def _update_volume_label(handle: PulseAsync):

            sink = await handle.sink_info(sink_idx)

            vol = round(sink.volume.value_flat * 100)

            if vol <= self.max_vol:
                GLib.idle_add(self.set_label_idle, self.results[sink.mute][vol])
                await asyncio.sleep(0.15)

        # if we start with volume level, that's more than 1.0 (100)
        GLib.idle_add(self.set_label_idle, self.results[0][100])
        await _update_volume_label(handle)

        async for event in handle.subscribe_events("sink"):
            if event.index == sink_idx and event.t == "change":
                await _update_volume_label(handle)


    async def _consume_user_events(self, handle: PulseAsync):

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
                tlisten = grp.create_task(self._listen_sink_events(pulse))
                tconsume = grp.create_task(self._consume_user_events(pulse))




class BarWidgetI3KeyboardLayout(I3ListenerMixin, RewriteMixin, BarWidget):
    def __init__(self,
                 rewrite: dict[str, str],
                 label_format: str,
                 i3_conn: Connection):
        super().__init__(0,
                         label_format,
                         None,
                         rewrite=rewrite,
                         i3_conn=i3_conn)

    async def run(self):
        for i3_i in await (await self.get_i3_conn()).get_inputs():
            if i3_i.xkb_active_layout_name is not None:
                self._push_layout(i3_i.xkb_active_layout_name)
                break

        async def _callback_kb_layout(_, event: InputEvent):
            evinput = event.input
            if evinput.type == "keyboard":
                if self._last_value != evinput.xkb_active_layout_name:
                    self._last_value = evinput.xkb_active_layout_name
                    self._push_layout(evinput.xkb_active_layout_name)

        (await self.get_i3_conn()).on(Event.INPUT, _callback_kb_layout)

    def _push_layout(self, raw_layout: str):
        if raw_layout not in self.cache:
            layout = self.rewrite(raw_layout)
            self.cache[raw_layout] = layout
        self.vupdate_label(layout=self.cache[raw_layout])


class BarWidgetI3Mode(I3ListenerMixin, RewriteMixin, BarWidget):

    def __init__(self,
                 rewrite: dict[str, str],
                 label_format: str,
                 always_show: bool,
                 i3_conn: Connection):
        super().__init__(0,
                         label_format,
                         None,
                         rewrite=rewrite,
                         i3_conn=i3_conn)
        self.always_show = always_show

    async def run(self):

        def _dispatch_mode(cur_mode: str):
            if self._last_value != cur_mode:
                self._last_value = cur_mode

                if cur_mode not in self.cache:
                    mode = self.rewrite(cur_mode)
                    self.cache[cur_mode] = mode
                self.vupdate_label(mode=self.cache[cur_mode])

            self.set_visible(
                not (self.always_show and cur_mode == "default")
            )

        _dispatch_mode("default")

        async def _callback_mode(_: Connection, event: ModeEvent):
            _dispatch_mode(event.change)

        (await self.get_i3_conn()).on(Event.MODE, _callback_mode)


class BarWidgetI3Window(I3ListenerMixin, RewriteMixin, BarWidget):

    def __init__(self,
                 rewrite: dict[str, str],
                 label_format: str,
                 i3_conn: Connection):
        super().__init__(0,
                         label_format,
                         None,
                         rewrite=rewrite,
                         i3_conn=i3_conn)

    async def run(self):
        self.set_visible(False)
        conn = await self.get_i3_conn()

        def _dispatch_con(con: Con):

            if con is not None and con:
                win_name = None
                if con.name is not None:
                    win_name = con.name
                elif con.app_id is not None:
                    win_name = con.app_id

                if win_name is not None:
                    self.set_visible(True)

                    if self._last_value != win_name:
                        self._last_value = win_name
                        if win_name not in self.cache:
                            self.cache[win_name] = self.rewrite(win_name)
                        self.vupdate_label(title=self.cache[win_name])
                else:
                    self.set_visible(False)
            else:
                self.set_visible(False)

        # Find the focused window title, if any, on start
        tree = await conn.get_tree()
        _dispatch_con(tree.find_focused())

        async def _callback_window(_: Connection, event: WindowEvent):
            if event.change == "focus":
                _dispatch_con(event.container)
            elif event.change == "close":
                # Find the focused window title after (possibly the last open)
                # window is closed
                tree = await conn.get_tree()
                _dispatch_con(tree.find_focused())

        conn.on(Event.WINDOW_FOCUS, _callback_window)
        conn.on(Event.WINDOW_CLOSE, _callback_window)


class BarWidgetI3Scratchpad(I3ListenerMixin, BarWidget):
    def __init__(self, label_format: str, always_show: bool, i3_conn: Connection):
        super().__init__(0, label_format, i3_conn=i3_conn)
        self.always_show = always_show

    async def run(self):

        conn = await self.get_i3_conn()

        def _dispatch_scratchpad(con: Con):
            num = 0

            if con is not None:
                if (scrpad := con.scratchpad()) is not None:
                    num = len(scrpad.nodes) + len(scrpad.floating_nodes)
                    if self._last_value != num:
                        self._last_value = num
                        self.vupdate_label(count=num)

            self.set_visible(not (self.always_show and num == 0))

        async def _callback_scratchpad(*_):
            _dispatch_scratchpad(await conn.get_tree())

        _dispatch_scratchpad(await conn.get_tree())

        conn.on(Event.WINDOW_MOVE, _callback_scratchpad)


class BarWidgetExecTail(BarWidget):

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
                            self.vupdate_label(**json.loads(line))
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


class BarWidgetExecRepeat(BarWidget):
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
            self.vupdate_label(**json.loads(line))
        self.proc = None

    def stop(self):
        if self.proc is not None and self.proc.returncode is None:
            self.proc.terminate()
        super().stop()


class I3WorkspaceButton(I3ListenerMixin, BarWidget):
    def __init__(self, name: str, label: str, i3_conn: Connection):
        super().__init__(0, None, i3_conn=i3_conn)
        self.set_name(name)
        self.set_label(label)
        self.add_css_class("workspace")
        self.onclick(1, CallableAction(self.switch_ws))
        self.aio_loop = asyncio.get_running_loop()

    async def _switch_ws_async(self, name: str):
        i3_conn = await self.get_i3_conn()
        return await i3_conn.command("workspace " + name)

    def switch_ws(self):
        name = self.get_name()
        self.aio_loop.call_soon_threadsafe(asyncio.ensure_future,
                                           self._switch_ws_async(name))


class BarWidgetI3Workspaces(I3ListenerMixin,
                            RewriteMixin,
                            Gtk.ScrolledWindow,
                            BarWindgetInterface):

    MAX_WORKSPACES = 20
    MAX_SCROLL_SPEED = 100

    def __init__(self,
                 rewrite: dict[str, str],
                 i3_conn: Connection,
                 scroll_width: int,
                 scroll_speed: int,
                 max_workspaces: int,
                 always_show: list[str] | None = None,
                 **kwargs):
        super().__init__(i3_conn=i3_conn, rewrite=rewrite)

        self.wsid_map: dict[str, int] = {}
        self.ws_button_map: dict[str, I3WorkspaceButton] = {}

        self.always_show = []
        if always_show is not None:
            self.always_show.extend([str(name) for name in always_show])

        self.max_workspaces = max(1, min(max_workspaces, self.MAX_WORKSPACES))

        self.set_propagate_natural_height(True)
        self.set_has_frame(False)
        self.set_kinetic_scrolling(False)

        self.box = Gtk.Box.new(Gtk.Orientation.HORIZONTAL, 0)

        self.cache: dict[str, Any] = {}
        self.viewport = None
        self.cur_focus = None
        self.prev_focus = None
        self.i3_conn = None

        if scroll_width > 0:
            scroll_speed = max(1, min(scroll_speed, self.MAX_SCROLL_SPEED))
            self.set_min_content_width(scroll_width)
            self.set_size_request(scroll_width, -1)
            self.set_policy(Gtk.PolicyType.EXTERNAL, Gtk.PolicyType.NEVER)
            self.viewport = Gtk.Viewport.new()
            self.viewport.set_child(self.box)

            self.h_adj = self.get_hadjustment()

            def _scroll(ctrl, _, direction):
                self.h_adj.set_value(
                    self.h_adj.get_value() + (direction * scroll_speed)
                )

            scroll_ctrl = Gtk.EventControllerScroll.new(
                Gtk.EventControllerScrollFlags.VERTICAL
            )
            scroll_ctrl.connect("scroll", _scroll)
            self.viewport.add_controller(scroll_ctrl)
            self.viewport.set_scroll_to_focus(False)
            self.set_child(self.viewport)
        else:
            self.set_propagate_natural_width(True)
            self.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.NEVER)
            self.set_child(self.box)

    def scroll_into_view(self, widget: Gtk.Widget):
        if self.viewport is not None:
            self.viewport.scroll_to(widget)


    def rewrite(self, text: str) -> str:
        if text not in self.cache:
            self.cache[text] = super().rewrite(text)
        return self.cache[text]


    def set_empty_idle(self, child: Gtk.Widget) -> None:
        child.set_visible(False)
        child.remove_css_class("focused")
        child.remove_css_class("previous")
        child.remove_css_class("urgent")
        return GLib.SOURCE_REMOVE

    def set_focused_idle(self, child: Gtk.Widget) -> None:
        child.add_css_class("focused")
        child.set_visible(True)
        self.scroll_into_view(child)
        return GLib.SOURCE_REMOVE

    def set_urgent_idle(self, child: Gtk.Widget) -> None:
        if child.has_css_class("urgent"):
            child.remove_css_class("urgent")
        else:
            child.add_css_class("urgent")
            child.set_visible(True)
            self.scroll_into_view(child)
        return GLib.SOURCE_REMOVE

    def add_css_class_idle(self, child: Gtk.Widget, css_class: str) -> None:
        child.add_css_class(css_class)
        return GLib.SOURCE_REMOVE


    def remove_css_class_idle(self, child: Gtk.Widget, css_class: str) -> None:
        child.remove_css_class(css_class)
        return GLib.SOURCE_REMOVE

    def set_name_idle(self, child: Gtk.Widget, name: str) -> None:
        child.set_name(name)
        return GLib.SOURCE_REMOVE

    def set_label_idle(self, child: Gtk.Widget, label: str) -> None:
        child.set_label(label)
        return GLib.SOURCE_REMOVE


    def dispatch_ws(self,
                    name: str,
                    old_focus_name: str,
                    action: str,
                    wsid: int) -> None:

        child = None
        old_name = None
        label = self.rewrite(name)

        if self.wsid_map.get(name, -1) < 0 and wsid >= 0:
            if action == "rename":
                for _name, _wsid in self.wsid_map.items():
                    if _wsid == wsid:
                        old_name = _name
                        break

                if old_name in self.wsid_map:
                    self.wsid_map[name] = self.wsid_map.pop(old_name)
                else:
                    self.wsid_map[name] = wsid

        if old_name in self.ws_button_map:
            child = self.ws_button_map[old_name]
            GLib.idle_add(self.set_name_idle, child, name)
            GLib.idle_add(self.set_label_idle, child, label)
        elif name in self.ws_button_map:
            child = self.ws_button_map[name]

            match action:
                case "empty":
                    if name not in self.always_show:
                        GLib.idle_add(self.set_empty_idle, child)
                    else:
                        GLib.idle_add(self.remove_css_class_idle, child, "focused")
                    GLib.idle_add(self.remove_css_class_idle, child, "urgent")
                case "focus":
                    if self.cur_focus in self.ws_button_map:
                        GLib.idle_add(self.remove_css_class_idle, self.ws_button_map[self.cur_focus], "focused")
                        GLib.idle_add(self.add_css_class_idle, self.ws_button_map[old_focus_name], "previous")

                    if self.prev_focus in self.ws_button_map:
                        GLib.idle_add(self.remove_css_class_idle, self.ws_button_map[self.prev_focus], "previous")

                    GLib.idle_add(self.set_focused_idle, child)

                    self.prev_focus, self.cur_focus = self.cur_focus, name
                case "urgent":
                    GLib.idle_add(self.set_urgent_idle, child)
                case _:
                    pass
        elif len(self.ws_button_map) < self.max_workspaces:
            child = I3WorkspaceButton(name, label, self.i3_conn)
            self.box.append(child)
            self.ws_button_map[name] = child
            if wsid >= 0:
                self.wsid_map[name] = wsid
        else:
            logging.error("refusing to track more than %d workspaces", self.max_workspaces)

        return GLib.SOURCE_REMOVE

    async def run(self):

        self.i3_conn = await self.get_i3_conn()

        async def _callback_workspaces(_, event: WorkspaceEvent):
            if event.change not in ["move", "restore", "reload"]:
                prev_name = None
                if event.old is not None:
                    prev_name = event.old.name

                self.dispatch_ws(event.current.name,
                    prev_name,
                    event.change,
                    event.current.id,
                )

        existing_ws = {}
        for ws in await self.i3_conn.get_workspaces():
            existing_ws[ws.name] = (ws.ipc_data["id"], ws.focused, ws.urgent)

        for name in self.always_show:
            wsid, *_ = existing_ws.get(name, (-1, False, False))

            if name is not None:
                self.dispatch_ws(name, None, "init", wsid)

        actions = ["init", "focus", "urgent"]
        for name, (wsid, focus, urgent) in existing_ws.items():
            for action in compress(actions, [True, focus, urgent]):
                self.dispatch_ws(name, None, action, wsid)

        self.i3_conn.on(Event.WORKSPACE, _callback_workspaces)


class PlayerctlButton(BarWidget):
    def __init__(self, name: str, label: str, label_format: str | None = None):
        super().__init__(0, label_format)

        self.initial_label = label
        self.set_name(name)
        self.set_label(label)
        self.add_css_class("playerctl-button")


    def reset_label(self):
        self.set_label(self.initial_label)
        # self.onclick(1, CallableAction(self.switch_ws))



class BarWidgetPlayerCtl(Gtk.Box):

    MAX_SCROLL_SPEED = 100

    SUPPORTED_MODULES = [

                "next",
                "play_pause",
                "previous",
                "seek_back",
                "seek_forward",
                "title",
                "volume",
                "time"

                ]



    def __init__(self, modules: list[dict[str, Any]],
                       always_show: bool = False,
                       tick_ms: int = 500,
                       **kwargs):
        super().__init__(**kwargs)
        self.set_name("playerctl")

        self.ticker_direction = 0

        self.cur_total_time = 0

        self.formatter = OptionalFormatter()

        self.manager = Playerctl.PlayerManager()

        self.player = None

        self.player_ready = asyncio.Event()

        self._last_time = 0.0

        self._run = True

        self.aio_loop = None

        self.tick = min(max(tick_ms, 10), 2000) / 1000

        self.always_show = always_show

        self.box_title = None
        self.btn_next = None
        self.btn_play_pause = None
        self.btn_previous = None
        self.btn_previous = None
        self.btn_seek_back = None
        self.btn_seek_forward = None
        self.btn_volume = None
        self.btn_time = None
        self.h_adj = None
        self.label_pause = None
        self.label_play = None
        self.scroll_view = None
        self.vol_labels = []
        self.ticker = False

        for module in modules:

            mod_type = module.get("type")
            if mod_type not in self.SUPPORTED_MODULES:
                continue

            match mod_type:
                case "play_pause":
                    if "label_play" in module and "label_pause" in module:
                        self.label_play = module["label_play"]
                        self.label_pause = module["label_pause"]
                        self.btn_play_pause = PlayerctlButton("playerctl-play-pause", self.label_play)
                        self.append(self.btn_play_pause)
                case "next":
                    if "label" in module:
                        self.btn_next = PlayerctlButton("playerctl-next", module["label"])
                        self.append(self.btn_next)
                case "previous":
                    if "label" in module:
                        self.btn_previous = PlayerctlButton("playerctl-previuos", module["label"])
                        self.append(self.btn_previous)
                case "seek_back":
                    if "label" in module:
                        self.btn_seek_back = PlayerctlButton("playerctl-seek-back", module["label"])
                        self.append(self.btn_seek_back)
                case "seek_forward":
                    if "label" in module:
                        self.btn_seek_forward = PlayerctlButton("playerctl-seek-forward", module["label"])
                        self.append(self.btn_seek_forward)
                case "time":
                    self.btn_time = PlayerctlButton("playerctl-time", module.get("label_empty"), module.get("label_format"))
                    self.append(self.btn_time)
                case "volume":
                    if "format" in module:
                        for vol in range(0, 101):
                            vol_label = self.formatter.format(module["format"], volume=vol)
                            self.vol_labels.append(vol_label)
                        self.btn_volume = PlayerctlButton("playerctl-volume", self.vol_labels[0])
                        self.append(self.btn_volume)
                case "title":

                    label_empty = module.get("label_empty", "")
                    scroll_speed = module.get("scroll_speed", 10)
                    scroll_width = module.get("scroll_width", 0)
                    label_format = module.get("label_format")

                    self.ticker = module.get("ticker", False)

                    self.label_title = PlayerctlButton("playerctl-title", label_empty, label_format)
                    self.scroll_view = Gtk.ScrolledWindow.new()
                    self.box_title = Gtk.Box.new(Gtk.Orientation.HORIZONTAL, 0)
                    self.box_title.append(self.label_title)

                    if scroll_width > 0:
                        scroll_speed = max(1, min(scroll_speed, self.MAX_SCROLL_SPEED))
                        self.scroll_view.set_min_content_width(scroll_width)
                        self.scroll_view.set_size_request(scroll_width, -1)
                        self.scroll_view.set_policy(Gtk.PolicyType.EXTERNAL, Gtk.PolicyType.NEVER)
                        viewport = Gtk.Viewport.new()
                        viewport.set_child(self.box_title)

                        self.h_adj = self.scroll_view.get_hadjustment()

                        def _scroll(ctrl, _, direction):
                            self.h_adj.set_value(
                                self.h_adj.get_value() + (direction * scroll_speed)
                            )

                        scroll_ctrl = Gtk.EventControllerScroll.new(
                            Gtk.EventControllerScrollFlags.VERTICAL
                        )

                        scroll_ctrl.connect("scroll", _scroll)
                        viewport.add_controller(scroll_ctrl)
                        viewport.set_scroll_to_focus(False)
                        self.scroll_view.set_child(viewport)
                    else:
                        self.scroll_view.set_propagate_natural_width(True)
                        self.scroll_view.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.NEVER)
                        self.scroll_view.set_child(self.box_title)

                    self.append(self.scroll_view)

        self.set_visible(self.always_show)

    def init_player(self, name):
        # choose if you want to manage the player based on the name
        # logging.debug("player appeared: %s", name.name)
        if name.name == "spotify_player":

            self.player = Playerctl.Player.new_from_name(name)

            if not self.player.props.can_control:
                raise RuntimeError("RTE")

            self.player.connect('playback-status', self.on_status)
            self.player.connect('metadata', self.on_metadata)
            self.player.connect('volume', self.on_volume)
            self.manager.manage_player(self.player)

            self.on_status(None, self.player.props.playback_status)
            self.on_volume(None, self.player.props.volume)
            self.on_metadata(None, self.player.props.metadata)


    def ticker_increment_idle(self):

        if self.ticker and self.h_adj is not None:

            curr_value = self.h_adj.get_value()

            if (curr_value + self.h_adj.get_page_size()) >= self.h_adj.get_upper():
                self.ticker_direction = -1
            elif curr_value <= 1:
                self.ticker_direction = 1

            if self.ticker_direction != 0:
                self.h_adj.set_value(curr_value + (self.ticker_direction * 10))

        return GLib.SOURCE_REMOVE


    def on_metadata(self, player, metadata):
        self.reset_time()

        keys = metadata.keys()

        # logging.debug(metadata)

        artist = 'Unknown Artist'
        if 'xesam:artist' in keys and metadata['xesam:artist']:
            artist = metadata['xesam:artist'][0]

        album = 'Unknown Album'
        if 'xesam:album' in keys and metadata['xesam:album']:
            album = metadata['xesam:album']

        title = 'Unknown Title'
        if 'xesam:title' in keys and metadata['xesam:title']:
            title = metadata['xesam:title']

        if 'mpris:length' in keys:
            self.cur_total_time = int(metadata['mpris:length'] / 10**6)

        GLib.idle_add(self.set_title_kw_idle, dict(artist=artist, album=album, title=title))

    def on_volume(self, player, volume: float):
        vol = int(volume)
        if self.vol_labels:
            vol_str = self.vol_labels[vol]
            GLib.idle_add(self.set_volume_idle, vol_str)


    def on_name_appeared(self, manager, name):
        logging.debug("playerctl: player name appeared: <%s>", name)
        self.init_player(name)

    def on_player_vanished(self, manager, player):
        self.player_ready.clear()
        self.on_status(None, Playerctl.PlaybackStatus.STOPPED)

    def reset_time(self):

        def _reset():
            self._last_time = 0

        if self.aio_loop is not None:
            self.aio_loop.call_soon_threadsafe(_reset)


    def on_status(self, player, status):
        self.reset_time()

        if status == Playerctl.PlaybackStatus.PLAYING:
            self.set_visible(True)
            self.player_ready.set()
            GLib.idle_add(self.set_play_idle, False)
        elif status == Playerctl.PlaybackStatus.PAUSED:
            self.player_ready.clear()
            GLib.idle_add(self.set_play_idle, True)
        else:
            self.set_visible(self.always_show)
            self.player_ready.clear()
            GLib.idle_add(self.set_title_kw_idle, None)
            GLib.idle_add(self.set_time_kw_idle, None)
            GLib.idle_add(self.set_volume_idle, None)


    def set_time_kw_idle(self, kwargs: dict[str, str] | None):
        if self.btn_time is not None:
            if kwargs is None:
                self.btn_time.reset_label()
            else:
                self.btn_time.vupdate_label(**kwargs)
        return GLib.SOURCE_REMOVE

    def set_play_idle(self, is_play: bool):
        if self.btn_play_pause is not None:
            if is_play:
                self.btn_play_pause.set_label(self.label_play)
            else:
                self.btn_play_pause.set_label(self.label_pause)

    def set_title_kw_idle(self, kwargs: dict[str, str] | None):
        if self.label_title is not None:
            if kwargs is None:
                self.label_title.reset_label()
            else:
                self.label_title.vupdate_label(**kwargs)
        return GLib.SOURCE_REMOVE

    def set_volume_idle(self, volume_str: str | None):
        if self.btn_volume is not None:
            if volume_str is None:
                self.btn_volume.reset_label()
            else:
                self.btn_volume.set_label(volume_str)
        return GLib.SOURCE_REMOVE

    async def watch_positon(self):

        while self._run:
            await self.player_ready.wait()

            self.reset_time()

            if self.player is not None:

                while self.player_ready.is_set():
                    if self.player.props.playback_status == Playerctl.PlaybackStatus.PLAYING:

                        raw_sec = int(self.player.get_position() / 10**6)

                        if self._last_time < raw_sec:
                            self._last_time = raw_sec

                            t_min, t_sec = divmod(raw_sec, 60)

                            t_tot_min, t_tot_sec = 0, 0 # TODO: move this to on_metadata
                            if self.cur_total_time > 0:
                                t_tot_min, t_tot_sec = divmod(self.cur_total_time, 60)

                            GLib.idle_add(self.set_time_kw_idle, dict(current=f"{t_min}:{t_sec:02d}", total=f"{t_tot_min}:{t_tot_sec:02d}"))

                        GLib.idle_add(self.ticker_increment_idle)

                        await asyncio.sleep(self.tick)
            await asyncio.sleep(1)

    async def run(self):
        self.aio_loop = asyncio.get_running_loop()

        self.manager.connect('name-appeared', self.on_name_appeared)
        self.manager.connect('player-vanished', self.on_player_vanished)

        for name in self.manager.props.player_names:
            self.init_player(name)

        async with asyncio.TaskGroup() as tg:
            task = tg.create_task(self.watch_positon())

    def stop(self):
        pass



class MehBarGUI(Gtk.ApplicationWindow):

    WIDGET_TYPE_MAP = {
        "cpu_percentage": {
            "class": BarWidgetCPUPercentage,
            "deps": ["psutil"],
            "unique": True,
            "kwargs": {
                "interval": 5,
            },
        },
        "temperature": {
            "class": BarWidgetTemperature,
            "unique": False,
            "kwargs": {
                "interval": 5,
                "max_temp": 100,
                "zone": 0,
            },
        },
        "datetime": {
            "class": BarWidgetDateTime,
            "unique": False,
            "kwargs": {
                "interval": 10,
            },
        },
        "exec_repeat": {
            "class": BarWidgetExecRepeat,
            "unique": False,
            "kwargs": {
                "interval": 5,
                "max_lps": 5,
            },
        },
        "exec_tail": {
            "class": BarWidgetExecTail,
            "unique": False,
            "kwargs": {
                "max_lps": 5,
            },
        },
        "memory": {
            "class": BarWidgetMemoryUsage,
            "deps": ["psutil"],
            "unique": True,
            "kwargs": {"interval": 11},
        },
        "disk": {
            "class": BarWidgetDiskUsage,
            "deps": ["psutil"],
            "unique": False,
            "kwargs": {"interval": 60, "path": "/"},
        },
        "playerctl": {
            "class": BarWidgetPlayerCtl,
            "deps": ["gi.repository/Playerctl"],
            "unique": True,
            "kwargs": {
                "always_show": True,

                "modules": [
                    {
                        "type": "previous",
                        "label": "\U000f0664"
                    },
                    {
                        "type": "seek_back",
                        "label": "\U000f11f9"
                    },
                    {
                        "type": "play_pause",
                        "label_play": "\U000f040d",
                        "label_pause": "\U000f03e6"
                    },
                    {
                        "type": "seek_forward",
                        "label": "\U000f11f8"
                    },
                    {
                        "type": "next",
                        "label": "\U000f0662"
                    },
                    {
                        "type": "title",
                        "label_empty": "-----",
                        "ticker": True,
                        "scroll_speed": 10,
                        "scroll_width": 128,
                        "label_format": "{artist} - {album} - {title}"
                    },
                    {
                        "type": "time",
                        "label_empty": "--:--",
                        "label_format": "{current}/{total}"
                    },

                ],
          },
        },
        "pulseaudio_volume": {
            "class": BarWidgetPulseVolume,
            "deps": ["pulsectl_asyncio/PulseAsync"],
            "unique": True,
            "kwargs": {"max_vol": 100, "vol_delta": 20, "sink_name": "@DEFAULT_SINK@"},
        },
        "static": {
            "class": BarWidgetStatic,
            "unique": False,
        },
        "i3_kblayout": {
            "class": BarWidgetI3KeyboardLayout,
            "deps": ["i3ipc", "i3ipc.aio/Connection", "i3ipc/Event", "i3ipc/InputEvent", "i3ipc/Con"],
            "unique": True
        },
        "i3_mode": {
            "class": BarWidgetI3Mode,
            "deps": ["i3ipc", "i3ipc.aio/Connection", "i3ipc/Event", "i3ipc/ModeEvent", "i3ipc/Con"],
            "unique": True,
            "kwargs": {"always_show": True},
        },
        "i3_scratchpad": {
            "class": BarWidgetI3Scratchpad,
            "deps": ["i3ipc", "i3ipc.aio/Connection", "i3ipc/Event", "i3ipc/WindowEvent", "i3ipc/Con"],
            "unique": True,
            "kwargs": {"always_show": True},
        },
        "i3_workspaces": {
            "class": BarWidgetI3Workspaces,
            "deps": ["i3ipc", "i3ipc.aio/Connection", "i3ipc/Event", "i3ipc/WorkspaceEvent", "i3ipc/Con"],
            "unique": True,
            "kwargs": {
                "always_show": ["1", "2", "3", "4"],
                "scroll_width": 0,
                "scroll_speed": 10,
                "max_workspaces": 10,
            },
        },
        "i3_window": {
            "class": BarWidgetI3Window,
            "deps": ["i3ipc", "i3ipc.aio/Connection", "i3ipc/Event", "i3ipc/WindowEvent", "i3ipc/Con"],
            "unique": True,
            "kwargs": {"always_show": True},
        },
        "wifi_signal": {
            "class": BarWidgetWifiSignal,
            "deps": [],
            "unique": True,
            "kwargs": {"interval": 10},
        },
    }

    DEFAULT_CONFIG = {
        "widgets": {
            "start": [
                {
                    "type": "i3_workspaces",
                    "rewrite": {},
                },
                {
                    "type": "temperature",
                    "ramp": ["\uf2cb", "\uf2c9", "\uf2c8", "\uf2c7"],
                    "label_format": "{ramp} {celsius}\u00b0C",
                },
                {
                    "type": "cpu_percentage",
                    "label_format": "\uf4bc {percent}%",
                },
                {
                    "type": "memory",
                    "label_format": "\U000f07af {percent}%",
                },
                {
                    "type": "disk",
                    "label_format": "\uf0a0 {used_gib} GiB",
                },
                {
                    "type": "static",
                    "classes": ["separator"],
                    "label_format": "\U000f01d9",
                },
                {
                    "type": "i3_scratchpad",
                    "label_format": "\U000f0ab6 {count}",
                },
                {
                    "type": "i3_mode",
                    "label_format": "{mode}",
                    "rewrite": {"resize": "\U000f0a68", "default": "\ueb7f"},
                },
                {
                    "type": "i3_window",
                    "label_format": "\ueb7f {title}",
                    "rewrite": {"foot": "Terminal", "footclient": "Terminal"},
                },
            ],
            "center": [
             {
                    "type": "playerctl",

                },

                # {
                #     "name": "testexec",
                #     "class": BarWidgetExecTail,
                #     "cmdline": ["/home/meh/testexec.sh"],
                #     "max_lps": 5,
                #     "label_format": "\uf489 {testvalue}",
                # },
                # {
                #     "name": "texexec2",
                #     "class": BarWidgetExecRepeat,
                #     "cmdline": ["/home/meh/testexec2.sh"],
                #     "interval": 3,
                #     "max_lps": 5,
                #     "label_format": "\uf489 {testvalue}",
                # }
            ],
            "end": [
                {
                    "type": "i3_kblayout",
                    "label_format": "{layout}",
                    "rewrite": {
                        "(?i)english.*": "US",
                        "(?i)ukrainian.*": "UA",
                        "(?i)german.*": "DE",
                    },
                },
                {
                    "type": "wifi_signal",
                    "iface": "wlan0",
                    "ramp": [
                        "\U000f092e",
                        "\U000f091f",
                        "\U000f0925",
                        "\U000f0928",
                    ],
                    "label_format": "{ramp} {signal}%",
                },
                {
                    "type": "pulseaudio_volume",
                    "ramp": [
                        "\U000f075f",
                        "\U000f057f",
                        "\U000f0580",
                        "\U000f0580",
                        "\U000f057e",
                    ],
                    "label_format": "{ramp} {percent}%",
                },
                {
                    "type": "datetime",
                    "datetime_format": "%H:%M",
                    "label_format": "\U000f0150 {datetime}",
                },
            ],
        }
    }

    css = b"""

        window.background {
            background: unset;
        }


        box {
            background-color: #efdddd;
        }


        .bar-widget {
            font-family: "RobotoMono Nerd Font";
            font-size: 12pt;
            padding: 0px 5px 0px 5px;
        }

        #end {
            padding-right: 8px;
        }

        #start {
            padding-left: 8px;
        }


        #i3_kblayout {
            font-size: 11pt;
            outline: 1px solid;
            outline-offset: -2px;
            border-radius: 8px;
            padding: 0px 8px 0px 8px;
            margin: 1px 5px 1px 5px;
        }

        .workspace {
            padding: 0px 8px 0px 8px;
        }

        .urgent {
            background-color: red;
        }

        .focused {
            background-color: #aaa;
        }

        .previous {
            background-color: #daa;
        }

        #playerctl {
            background-color: #fee;
        }

        #playerctl-previuos, #playerctl-play-pause, #playerctl-next {
            padding: 0 6px 0 6px;
        }

        """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        homo = True

        self.i3_conn = None
        self.widget_list = []
        self._unique_widget_types = set()
        self.task_ref_set = set()

        self.app_ = self.get_application()
        self.set_default_size(self.get_primary_mon_width(), 24)

        Gtk4LayerShell.init_for_window(self)
        Gtk4LayerShell.set_layer(self, Gtk4LayerShell.Layer.TOP)
        Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.BOTTOM, True)
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.BOTTOM, 8)
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.TOP, 8)
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.LEFT, 8)
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.RIGHT, 8)
        Gtk4LayerShell.auto_exclusive_zone_enable(self)

        style_provider = Gtk.CssProvider()
        style_provider.load_from_data(self.css)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display().get_default(),
            style_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        self.main_box = Gtk.Box.new(Gtk.Orientation.HORIZONTAL, 0)
        self.main_box.set_homogeneous(homo) # TODO: Make configurable
        self.start_box = Gtk.Box.new(Gtk.Orientation.HORIZONTAL, 0)
        self.start_box.set_halign(Gtk.Align.START)
        self.start_box.set_valign(Gtk.Align.CENTER)
        self.center_box = Gtk.Box.new(Gtk.Orientation.HORIZONTAL, 0)
        self.center_box.set_valign(Gtk.Align.CENTER)
        if homo:
            self.center_box.set_halign(Gtk.Align.CENTER)
        else:
            self.center_box.set_halign(Gtk.Align.START)
            self.center_box.set_hexpand(True)

        self.end_box = Gtk.Box.new(Gtk.Orientation.HORIZONTAL, 0)
        self.end_box.set_halign(Gtk.Align.END)
        self.end_box.set_valign(Gtk.Align.CENTER)
        self.main_box.set_valign(Gtk.Align.CENTER)
        self.main_box.append(self.start_box)
        self.main_box.append(self.center_box)
        self.main_box.append(self.end_box)


        self.set_child(self.main_box)

    @staticmethod
    def get_primary_mon_width() -> int:
        display = Gdk.Display.get_default()
        width = 0
        for monitor in display.get_monitors():
            geometry = monitor.get_geometry()
            width = (geometry.y + geometry.width) - geometry.y
            if width > 0:
                break
        return width

    def new_widget_for(self, *_, **kwargs) -> BarWidget:

        widget = None

        if (w_type := kwargs.pop("type", None)) is None:
            raise BarConfigError("missing 'type' field in widget config")

        missing_deps = set()
        for dep in self.WIDGET_TYPE_MAP[w_type].get("deps", []):
            split = dep.split("/")
            if len(split) == 2:
                mod_name, obj_name = split
                if mod_name not in sys.modules or not hasattr(sys.modules[mod_name], obj_name):
                    missing_deps.add(dep)
            elif len(split) == 1:
                if split[0] not in sys.modules:
                     missing_deps.add(dep)

        if missing_deps:
            raise BarConfigError(f"missing dependencies: {', '.join(missing_deps)}")

        if w_type not in self.WIDGET_TYPE_MAP:
            available = ", ".join(self.WIDGET_TYPE_MAP.keys())
            raise BarConfigError(f"unknown widget type: {w_type}. Available: {available}")


        w_unique = self.WIDGET_TYPE_MAP[w_type].get("unique", False)

        if w_unique:
            if w_type in self._unique_widget_types:
                raise BarConfigError(f"windget of type '{w_type}' must be unique")
            self._unique_widget_types.add(w_type)


        w_class: Callable = self.WIDGET_TYPE_MAP[w_type]["class"]

        w_css_classes = kwargs.pop("classes", [])
        w_id = kwargs.pop("id", None)

        _kwargs: dict[str, Any] = self.WIDGET_TYPE_MAP[w_type].get(
            "kwargs", {}
        )
        overlay_dict_r(_kwargs, kwargs)

        w_max_width = _kwargs.pop("max_width", 0)
        w_width = _kwargs.pop("width", 0)

        w_class_args = set()

        for name, param in signature(w_class).parameters.items():

            w_class_args.add(name)

            if name != "kwargs":
                if name == "i3_conn":
                    _kwargs[name] = self.i3_conn

                if param.default == Parameter.empty:
                    assert (
                        name in _kwargs
                    ), f"No key {name} for widget of type {w_type}"

        del_args = set()

        for act_arg in _kwargs.keys():
            if act_arg not in w_class_args:
                del_args.add(act_arg)

        for del_arg in del_args:
            del _kwargs[del_arg]

        w_class_args.clear()
        del_args.clear()

        if _kwargs.get("interval", 1) < 1:
            raise BarConfigError("the value of 'interval' field in widget config must not be less than 1")

        widget = w_class(**_kwargs)

        if w_id is not None:
            widget.set_name(w_id)
        elif w_unique:
            widget.set_name(w_type)

        for css_class in w_css_classes:
            widget.add_css_class(css_class)

        if w_width > 0:
            widget.set_width_chars(w_width)

        if w_max_width > 0:
            widget.set_max_width_chars(w_max_width)

        return widget

    def init_widgets(self):
        for section in ["start", "center", "end"]:
            box = getattr(self, f"{section}_box")

            if section in self.DEFAULT_CONFIG["widgets"]:
                for kwargs in self.DEFAULT_CONFIG["widgets"][section]:
                    try:
                        widget = self.new_widget_for(**kwargs)
                    except BarConfigError as ex:
                        w_type = kwargs.get("type", "unknown")
                        i_id = kwargs.get("id", "unknown")
                        logging.error("disabling widget of type '%s', ID '%s', reason: %s", w_type, i_id, ex)
                    else:
                        self.widget_list.append(widget)
                        box.append(widget)
            else:
                box.set_visible(False)

    async def windget_runner(self, widget: Gtk.Widget, name: str | None = None):
        try:
            await widget.run()
        except Exception as ex:
            widget.stop()
            widget.set_visible(False)
            logging.error("disabling widget: %s, reason: runtime error: %s", name, ex)
            # raise


    async def run_widgets(self):
        async with asyncio.TaskGroup() as grp:
            for widget in self.widget_list:
                name = widget.get_name()
                task = grp.create_task(self.windget_runner(widget, name), name=name)
                self.task_ref_set.add(task)
                task.add_done_callback(self.task_ref_set.discard)

    async def init_and_run_widgets(self):
        self.init_widgets()
        await self.run_widgets()


class MehBar(Gtk.Application):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.win = None

    def do_activate(self, *args, **kwargs):
        if (active_window := self.get_active_window()) is not None:
            active_window.present()
        else:
            self.win = MehBarGUI(application=self)

            self.win.init_widgets()
            t_module_worker = threading.Thread(
                target=lambda: asyncio.run(self.win.run_widgets())
            )
            t_module_worker.daemon = True
            t_module_worker.start()
            self.win.present()


if __name__ == "__main__":
    app = MehBar(
        application_id="com.github.mehsayer.mehbar",
        flags=Gio.ApplicationFlags.FLAGS_NONE,
    )
    # GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, app.quit)
    app.run([])
