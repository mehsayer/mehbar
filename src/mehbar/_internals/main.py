#!/usr/bin/env python3
from __future__ import annotations

from typing import Any
from collections.abc import Callable
from ctypes import CDLL
from inspect import Parameter, signature

import asyncio
import logging
import os
import signal
import sys
import threading

EXC_INFO = os.getenv("MEHBAR_LOG_EXCEPTIONS") is not None

import gi

# GSK_RENDERER=cairo GDK_BACKEND=wayland

cdll_failed = set()

for soname in ["libgtk4-layer-shell.so.0", "libgtk4-layer-shell.so.0"]:
    try:
        CDLL(soname)
        break
    except OSError:
        cdll_failed.add(soname)
    else:
        cdll_failed.clear()

if cdll_failed:
    logging.critical(
        "failed to load GTK4 layer shell library, tried: %s", ", ".join(cdll_failed)
    )

    sys.exit(1)
try:
    gi.require_version("Gtk", "4.0")
    gi.require_version("Gtk4LayerShell", "1.0")
    from gi.repository import Gtk, Gdk, Gio, GLib
    from gi.repository import Gtk4LayerShell
except ValueError:
    logging.critical("failed to select required GTK4 version", exc_info=EXC_INFO)
    sys.exit(1)

try:
    gi.require_version("Playerctl", "2.0")
    from gi.repository import Playerctl
except ImportError:
    logging.critical("failed to select required Playerctl version", exc_info=EXC_INFO)
    sys.exit(1)


from mehbar._widgets import (
    WidgetCPUPercentage,
    WidgetTemperature,
    WidgetDateTime,
    WidgetStatic,
    WidgetExecTail,
    WidgetExecRepeat,
    WidgetMemoryUsage,
    WidgetPlayerCtl,
    WidgetPulseVolume,
    WidgetDiskUsage,
    WidgetI3Mode,
    WidgetI3Window,
    WidgetI3Scratchpad,
    WidgetI3Workspaces,
    WidgetI3KeyboardLayout,
    WidgetWifiSignal,
    WidgetWired,
    WidgetFanSpeed,
    WidgetNetworkRate,
    WidgetSession,
    WidgetBattery,
    WidgetBacklight,
    WidgetBluetooth,
)

from mehbar.widgets import Widget
from mehbar.exceptions import BarConfigError
from mehbar.tools import overlay_dict_r



def get_primary_mon_width() -> int:
    display = Gdk.Display.get_default()
    width = 0
    for monitor in display.get_monitors():
        geometry = monitor.get_geometry()
        width = (geometry.y + geometry.width) - geometry.y
        if width > 0:
            break
    return width


class MehBarGUI(Gtk.ApplicationWindow):
    WIDGET_TYPE_MAP = {
        "cpu_percentage": {
            "class": WidgetCPUPercentage,
            "deps": ["psutil"],
            "unique": True,
            "kwargs": {
                "interval": 5,
            },
        },
        "temperature": {
            "class": WidgetTemperature,
            "unique": False,
            "kwargs": {
                "interval": 5,
                "max_temp": 100,
                "source": 0,
            },
        },
        "fan_speed": {
            "class": WidgetFanSpeed,
            "unique": False,
            "kwargs": {
                "interval": 5,
                "max_speed": 5000,
                "source": None,
            },
        },
        "datetime": {
            "class": WidgetDateTime,
            "unique": False,
            "kwargs": {
                "interval": 10,
            },
        },
        "exec_repeat": {
            "class": WidgetExecRepeat,
            "unique": False,
            "kwargs": {
                "interval": 5,
                "max_lps": 5,
            },
        },
        "exec_tail": {
            "class": WidgetExecTail,
            "unique": False,
            "kwargs": {
                "max_lps": 5,
            },
        },
        "memory": {
            "class": WidgetMemoryUsage,
            "deps": ["psutil"],
            "unique": True,
            "kwargs": {"interval": 12},
        },
        "disk": {
            "class": WidgetDiskUsage,
            "deps": ["psutil"],
            "unique": False,
            "kwargs": {"interval": 60, "path": "/"},
        },
        "session": {
            "class": WidgetSession,
            "deps": ["psutil"],
            "unique": False,
            "kwargs": {"interval": 5},
        },
        "battery": {
            "class": WidgetBattery,
            "deps": ["psutil"],
            "unique": False,
            "kwargs": {"interval": 5},
        },
        "backlight": {
            "class": WidgetBacklight,
            "unique": False,
            "kwargs": {"interval": 5, "driver": "acpi", "step": 10},
        },
        "bluetooth": {
            "class": WidgetBluetooth,
            "unique": True,
            "kwargs": {"interval": 5},
        },
        "playerctl": {
            "class": WidgetPlayerCtl,
            "deps": ["gi.repository/Playerctl"],
            "unique": True,
            "kwargs": {
                "always_show": True,
                "player_names": [],
                "modules": [
                    {"type": "previous", "label": "\U000f0664"},
                    {"type": "seek_back", "label": "\U000f11f9"},
                    {"type": "shuffle", "label_on": "\uf074", "label_off": "\uf0cb"},
                    {
                        "type": "play_pause",
                        "label_play": "\U000f040d",
                        "label_pause": "\U000f03e6",
                    },
                    {"type": "seek_forward", "label": "\U000f11f8"},
                    {"type": "next", "label": "\U000f0662"},
                    {
                        "type": "title",
                        "label_empty": "-----",
                        "ticker": True,
                        "scroll_speed": 10,
                        "scroll_width": 128,
                        "label_format": "{artist} - {album} - {title}",
                    },
                    {
                        "type": "time",
                        "label_empty": "--:--",
                        "label_format": "{current}/{total}",
                    },
                ],
            },
        },
        "pulseaudio_volume": {
            "class": WidgetPulseVolume,
            "deps": ["pulsectl_asyncio/PulseAsync"],
            "unique": True,
            "kwargs": {"max_vol": 100, "vol_delta": 20, "sink_name": "@DEFAULT_SINK@"},
        },
        "static": {
            "class": WidgetStatic,
            "unique": False,
        },
        "i3_kblayout": {
            "class": WidgetI3KeyboardLayout,
            "deps": [
                "i3ipc",
                "i3ipc.aio/Connection",
                "i3ipc/Event",
                "i3ipc/InputEvent",
                "i3ipc/Con",
            ],
            "unique": True,
        },
        "i3_mode": {
            "class": WidgetI3Mode,
            "deps": [
                "i3ipc",
                "i3ipc.aio/Connection",
                "i3ipc/Event",
                "i3ipc/ModeEvent",
                "i3ipc/Con",
            ],
            "unique": True,
            "kwargs": {"always_show": True},
        },
        "i3_scratchpad": {
            "class": WidgetI3Scratchpad,
            "deps": [
                "i3ipc",
                "i3ipc.aio/Connection",
                "i3ipc/Event",
                "i3ipc/WindowEvent",
                "i3ipc/Con",
            ],
            "unique": True,
            "kwargs": {"always_show": True},
        },
        "i3_workspaces": {
            "class": WidgetI3Workspaces,
            "deps": [
                "i3ipc",
                "i3ipc.aio/Connection",
                "i3ipc/Event",
                "i3ipc/WorkspaceEvent",
                "i3ipc/Con",
            ],
            "unique": True,
            "kwargs": {
                "always_show": ["1", "2", "3", "4"],
                "scroll_width": 0,
                "scroll_speed": 10,
                "max_workspaces": 10,
            },
        },
        "i3_window": {
            "class": WidgetI3Window,
            "deps": [
                "i3ipc",
                "i3ipc.aio/Connection",
                "i3ipc/Event",
                "i3ipc/WindowEvent",
                "i3ipc/Con",
            ],
            "unique": True,
            "kwargs": {"always_show": True},
        },
        "wifi": {
            "class": WidgetWifiSignal,
            "deps": [],
            "unique": True,
            "kwargs": {"interval": 1},
        },
        "wired": {
            "class": WidgetWired,
            "deps": [],
            "unique": True,
            "kwargs": {"interval": 1},
        },
        "network_rate": {
            "class": WidgetNetworkRate,
            "deps": [],
            "unique": False,
            "kwargs": {
                "interval": 1,
                "conv_map": {"Kb/s": 1024, "Mb/s": 1024**2, "b/s": 1},
            },
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
                    "label_format": "{ramp} {temp}\u00b0C",
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
                {"type": "playerctl", "player_names": ["spotify_player", "firefox"]},
                {
                    "type": "backlight",
                    "device": 13,
                    "label_format": "{level}% {ramp}",
                    "ramp": ["A", "B", "C"],
                },
                {
                    "type": "bluetooth",
                    "interval": 1,
                    "label_format": "{ramp}",
                    "ramp": ["OFF", "ON", "CONN"],
                },
                # {
                #     "type": "battery",
                #     "label_format": "{ramp} {percent}% {timeleft}",
                #     "interval": 1,
                #     "ramp": ["\U000f125e",
                #             "\U000f008e",
                #             "\U000f089f",
                #             "\U000f007b",
                #             "\U000f0086",
                #             "\U000f007d",
                #             "\U000f0088",
                #             "\U000f007f",
                #             "\U000f0089", "\U000f0081", "\U000f008a", "\U000f0079", "\U000f0085"
                #             ]
                # },
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
                    "type": "wired",
                    "iface": "eth0",
                    "ramp": [
                        "II",
                        "IA",
                        "AA",
                    ],
                    "backend": "unmanaged",
                    "label_format": "{ipv4} {ramp}",
                },
                {
                    "type": "wifi",
                    "iface": "wlan0",
                    "ramp": [
                        "\U000f092e",
                        "\U000f091f",
                        "\U000f0925",
                        "\U000f0928",
                    ],
                    "backend": "iwd",
                    "label_format": "{ramp} {ssid} {percentage}%",
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
                    "label_format": "\U000f0150 {datetime:%H:%M}",
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

        homo = False

        self.i3_conn = None
        self.widget_list = []
        self._unique_widget_types = set()
        self.task_ref_set = set()

        self.app_ = self.get_application()
        self.set_default_size(get_primary_mon_width(), 24)

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
        self.main_box.set_homogeneous(homo)  # TODO: Make configurable
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

    def new_widget_for(self, *_, **kwargs) -> Widget:

        widget = None

        if (w_type := kwargs.pop("type", None)) is None:
            raise BarConfigError("missing 'type' field in widget config")

        missing_deps = set()
        for dep in self.WIDGET_TYPE_MAP[w_type].get("deps", []):
            split = dep.split("/")
            if len(split) == 2:
                mod_name, obj_name = split
                if mod_name not in sys.modules or not hasattr(
                    sys.modules[mod_name], obj_name
                ):
                    missing_deps.add(dep)
            elif len(split) == 1:
                if split[0] not in sys.modules:
                    missing_deps.add(dep)

        if missing_deps:
            raise BarConfigError(f"missing dependencies: {', '.join(missing_deps)}")

        if w_type not in self.WIDGET_TYPE_MAP:
            available = ", ".join(self.WIDGET_TYPE_MAP.keys())
            raise BarConfigError(
                f"unknown widget type: {w_type}. Available: {available}"
            )

        w_unique = self.WIDGET_TYPE_MAP[w_type].get("unique", False)

        if w_unique:
            if w_type in self._unique_widget_types:
                raise BarConfigError(f"windget of type '{w_type}' must be unique")
            self._unique_widget_types.add(w_type)

        w_class: Callable = self.WIDGET_TYPE_MAP[w_type]["class"]

        w_css_classes = kwargs.pop("classes", [])
        w_id = kwargs.pop("id", None)

        _kwargs: dict[str, Any] = self.WIDGET_TYPE_MAP[w_type].get("kwargs", {})
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
                    assert name in _kwargs, f"No key {name} for widget of type {w_type}"

        del_args = set()

        for act_arg in _kwargs.keys():
            if act_arg not in w_class_args:
                del_args.add(act_arg)

        for del_arg in del_args:
            del _kwargs[del_arg]

        w_class_args.clear()
        del_args.clear()

        if _kwargs.get("interval", 1) < 1:
            raise BarConfigError(
                "the value of 'interval' field in widget config must not be less than 1"
            )

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
                        logging.error(
                            "disabling widget of type '%s', ID '%s', reason: %s",
                            w_type,
                            i_id,
                            ex,
                            exc_info=EXC_INFO,
                        )
                    else:
                        self.widget_list.append(widget)
                        box.append(widget)
            else:
                box.set_visible(False)

    async def _run_widget(self, widget: Gtk.Widget, name: str | None = None):
        try:
            await widget.run()
        except Exception as ex:
            widget.stop()
            widget.set_visible(False)
            logging.error(
                "disabling widget: %s, reason: %s", name, ex, exc_info=EXC_INFO
            )

    async def run_widgets(self):
        async with asyncio.TaskGroup() as grp:
            for widget in self.widget_list:
                name = widget.get_name()
                task = grp.create_task(self._run_widget(widget, name), name=name)
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


def entrypoint(argv):
    app = MehBar(
        application_id="com.github.mehsayer.mehbar",
        flags=Gio.ApplicationFlags.FLAGS_NONE,
    )
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, app.quit)
    app.run(argv)
