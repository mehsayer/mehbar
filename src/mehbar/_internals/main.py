#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import importlib
import logging
import os
import signal
import sys
from collections.abc import Callable
from ctypes import CDLL
from functools import partial
from inspect import Parameter, signature
from pathlib import Path
from threading import Thread
from typing import Any

try:
    import anyio
except ImportError:
    logging.critical("AnyIO (anyio) module is not found")
    sys.exit(1)
try:
    import gi
except ImportError:
    logging.critical("PyGObject (PyGObject) module is not found")
    sys.exit(1)

cdll_failed = set()

for soname in ["libgtk4-layer-shell.so.0", "libgtk4-layer-shell.so.1"]:
    try:
        CDLL(soname)
        cdll_failed.clear()
        break
    except OSError:
        cdll_failed.add(soname)

if cdll_failed:
    logging.critical(
        "failed to load GTK4 layer shell library, tried: %s", ", ".join(cdll_failed)
    )
    sys.exit(1)
try:
    gi.require_version("Gtk", "4.0")
    gi.require_version("Gdk", "4.0")
    gi.require_version("Gtk4LayerShell", "1.0")
except ValueError as ex:
    logging.critical(str(ex))
    sys.exit(1)

from gi.repository import Gdk, Gio, GLib, Gtk, Gtk4LayerShell

from mehbar._widgets import (
    WidgetBacklight,
    WidgetBattery,
    WidgetBluetooth,
    WidgetCPUPercentage,
    WidgetDateTime,
    WidgetDiskUsage,
    WidgetExecRepeat,
    WidgetExecTail,
    WidgetFanSpeed,
    WidgetI3KeyboardLayout,
    WidgetI3Mode,
    WidgetI3Scratchpad,
    WidgetI3Window,
    WidgetI3Workspaces,
    WidgetMemoryUsage,
    WidgetNetworkRate,
    WidgetPlayerCtl,
    WidgetPulseVolume,
    WidgetSession,
    WidgetStatic,
    WidgetTemperature,
    WidgetWifiSignal,
    WidgetWired,
)
from mehbar.exceptions import BarConfigError
from mehbar.tools import overlay_dict_r
from mehbar.widgets import Widget

# GSK_RENDERER=cairo GDK_BACKEND=wayland


def get_config_home() -> Path:
    if (cfg_home := os.getenv("XDG_CONFIG_HOME")) is None:
        cfg_home = Path.home() / ".config"
    else:
        cfg_home = Path(cfg_home)
    return cfg_home / "mehbar"


def load_config() -> dict[str, Any]:
    ret = {}

    cfg_home = get_config_home()

    import_map = {
        "tomli": "config.toml",
        "toml": "config.toml",
        "tomllib": "config.toml",
        "json5": "config.json5",
        "pyjson5": "config.json5",
        "jsonc": "config.jsonc",
        "json": "config.json",
    }

    if (envvar_parser := os.getenv("MEHBAR_CONFIG_PARSER_MODULE")) is not None:
        if envvar_parser in import_map:
            for drop_parser in list(import_map.keys()):
                if drop_parser != envvar_parser:
                    del import_map[drop_parser]

    tried = []
    failed = []

    for module_name, config_file_name in import_map.items():
        cfg_file = cfg_home / config_file_name

        if cfg_file.is_file():
            tried.append((cfg_file, module_name))
            try:
                parser = importlib.import_module(module_name)

                with open(cfg_file, "rb") as fhandle:
                    ret = parser.load(fhandle)

                if ret:
                    logging.debug("loaded '%s' using '%s'", cfg_file, module_name)
                    break
            except Exception as ex:
                failed.append((cfg_file, module_name, ex))

    if not ret:
        if failed:
            for cfg_file, module_name, ex in failed:
                logging.error(
                    "failed to load '%s' using '%s': %s",
                    cfg_file,
                    module_name,
                    ex,
                )
        elif not tried:
            logging.error("no configuration files found in '%s'", cfg_file.parent)
        else:
            for cfg_file, module_name in tried:
                logging.info("tried to load '%s' using '%s'", cfg_file, module_name)
        logging.critical("failed to load configuration")

    return ret


def load_css() -> bytes:
    ret = b"\n"

    css_file_path = get_config_home() / "style.css"

    try:
        with open(css_file_path, "rb") as fhandle:
            ret += fhandle.read()
    except Exception as ex:
        logging.error("unable to load CSS stylesheet from '%s': %s", css_file_path, ex)

    return ret


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
            "unique": True,
            "kwargs": {"interval": 12},
        },
        "disk": {
            "class": WidgetDiskUsage,
            "unique": False,
            "kwargs": {"interval": 60, "path": "/"},
        },
        "session": {
            "class": WidgetSession,
            "unique": False,
            "kwargs": {"interval": 5},
        },
        "battery": {
            "class": WidgetBattery,
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
            "unique": True,
            "kwargs": {
                "always_show": True,
                "player_names": [],
                "modules": [
                    {"type": "previous", "label": "prev"},
                    {"type": "seek_back", "label": "rw"},
                    {
                        "type": "shuffle",
                        "label_on": "shuffle",
                        "label_off": "no shuffle",
                    },
                    {
                        "type": "play_pause",
                        "label_play": "play",
                        "label_pause": "pause",
                    },
                    {"type": "seek_forward", "label": "ff"},
                    {"type": "next", "label": "next"},
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
            "unique": True,
            "kwargs": {"max_vol": 100, "vol_delta": 10, "sink_name": "@DEFAULT_SINK@"},
        },
        "static": {
            "class": WidgetStatic,
            "unique": False,
        },
        "i3_kblayout": {
            "class": WidgetI3KeyboardLayout,
            "unique": True,
        },
        "i3_mode": {
            "class": WidgetI3Mode,
            "unique": True,
            "kwargs": {"always_show": True},
        },
        "i3_scratchpad": {
            "class": WidgetI3Scratchpad,
            "unique": True,
            "kwargs": {"always_show": True},
        },
        "i3_workspaces": {
            "class": WidgetI3Workspaces,
            "unique": True,
            "kwargs": {
                "always_show": ["1", "2", "3", "4"],
                "scroll_width": 0,
                "scroll_speed": 10,
                "max_workspaces": 10,
                "rewrite": {},
            },
        },
        "i3_window": {
            "class": WidgetI3Window,
            "unique": True,
            "kwargs": {"always_show": True},
        },
        "wifi": {
            "class": WidgetWifiSignal,
            "unique": True,
            "kwargs": {"interval": 1},
        },
        "wired": {
            "class": WidgetWired,
            "unique": True,
            "kwargs": {"interval": 1},
        },
        "network_rate": {
            "class": WidgetNetworkRate,
            "unique": False,
            "kwargs": {
                "interval": 1,
                "conv_map": {"Kb/s": 1024, "Mb/s": 1024**2, "b/s": 1},
            },
        },
    }

    BASE_CSS = b"""
        window.background {
            background: unset;
        }

        box {
            background-color: #efdddd;
        }

        #end {
            padding-right: 8px;
        }

        #start {
            padding-left: 8px;
        }

        #playerctl-previuos, #playerctl-play-pause, #playerctl-next {
            padding: 0 8px 0 8px;
        }

         .workspace {
             padding: 0px 8px 0px 8px;
         }

         .bar-widget {
             font-size: 10pt;
             padding: 0px 6px 0px 6px;
             font-family: "Monospace";
         }
        """

    ANCHOR_MAP = {"top": Gtk4LayerShell.Edge.TOP, "bottom": Gtk4LayerShell.Edge.BOTTOM}
    LAYER_MAP = {
        "top": Gtk4LayerShell.Layer.TOP,
        "bottom": Gtk4LayerShell.Layer.BOTTOM,
    }
    MIN_HEIGHT = 24
    MIN_WIDTH = 256
    DEFAULT_ANCHOR = "top"
    DEFAULT_LAYER = "top"
    DEFAULT_GAPS = [0, 0]
    DEFAULT_HOMOGENOUS = True
    SECTION_NAMES = ["start", "center", "end"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.i3_conn = None
        self.widget_list = []
        self._unique_widget_types = set()

        self.config = load_config()

        self.bar_config = self.config.get("bar")

        if not self.bar_config or self.bar_config is None:
            raise RuntimeError("no valid configuration available")

        is_homogenous = self.bar_config.get("homogenous", self.DEFAULT_HOMOGENOUS)

        anchor = self.ANCHOR_MAP.get(
            self.bar_config.get("position", self.DEFAULT_ANCHOR), self.DEFAULT_ANCHOR
        )

        layer = self.LAYER_MAP.get(
            self.bar_config.get("layer", self.DEFAULT_LAYER), self.DEFAULT_LAYER
        )

        height = self.bar_config.get("height", self.MIN_HEIGHT)

        total_width = self.bar_config.get("width", 0)

        gaps = self.bar_config.get("gaps", self.DEFAULT_GAPS)

        if len(gaps) == 2:
            gaps *= 2
        elif len(gaps) != 4 or not all([gap >= 0 for gap in gaps]):
            gaps = [0, 0, 0, 0]

        if total_width <= self.MIN_WIDTH:
            total_width = get_primary_mon_width()

        self.set_default_size(total_width - (gaps[1] + gaps[3]), height)

        Gtk4LayerShell.init_for_window(self)
        Gtk4LayerShell.set_layer(self, layer)
        Gtk4LayerShell.set_anchor(self, anchor, True)
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.TOP, gaps[0])
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.LEFT, gaps[1])
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.BOTTOM, gaps[2])
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.RIGHT, gaps[3])
        Gtk4LayerShell.auto_exclusive_zone_enable(self)

        style_provider = Gtk.CssProvider()
        css_stylesheet = self.BASE_CSS + load_css()
        style_provider.load_from_data(css_stylesheet)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display().get_default(),
            style_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        self.main_box = Gtk.Box.new(Gtk.Orientation.HORIZONTAL, 0)
        self.main_box.set_homogeneous(is_homogenous)
        self.start_box = Gtk.Box.new(Gtk.Orientation.HORIZONTAL, 0)
        self.start_box.set_halign(Gtk.Align.START)
        self.start_box.set_valign(Gtk.Align.CENTER)
        self.center_box = Gtk.Box.new(Gtk.Orientation.HORIZONTAL, 0)
        self.center_box.set_valign(Gtk.Align.CENTER)

        if is_homogenous:
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

    def new_widget_for(self, w_name: str, **kwargs) -> Widget:

        widget = None

        if w_name in self.SECTION_NAMES:
            raise BarConfigError(f"name '{w_name}' is not allowed for widgets")

        if (w_type := kwargs.pop("type", None)) is None:
            raise BarConfigError("missing 'type' field in widget config")

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

        if _kwargs.get("interval", 0) < 0:
            raise BarConfigError(
                "the value of 'interval' parameter must not be less than 0"
            )

        widget = w_class(**_kwargs)

        if w_name is not None:
            widget.set_name(w_name)

        elif w_unique:
            widget.set_name(w_type)

        for css_class in w_css_classes:
            widget.add_css_class(css_class)

        if w_width > 0:
            widget.set_width_chars(w_width)

        if w_max_width > 0:
            widget.set_max_width_chars(w_max_width)

        return widget

    def _init_widgets(self):

        for section in ["start", "center", "end"]:
            box = getattr(self, f"{section}_box")

            if section in self.config:
                for w_name, kwargs in self.config[section].items():
                    try:
                        widget = self.new_widget_for(w_name, **kwargs)
                    except BarConfigError as ex:
                        w_type = kwargs.get("type", "unknown")
                        logging.error(
                            "disabling widget of type '%s', name '%s', reason: %s",
                            w_type,
                            w_name,
                            ex,
                        )
                    else:
                        self.widget_list.append(widget)
                        box.append(widget)
            else:
                box.set_visible(False)

    async def _run_widget(self, widget: Gtk.Widget, name: str | None = None):
        try:
            await widget.run_wrapper()
        except Exception as ex:
            widget.shutdown()
            widget.set_visible(False)
            widget.get_parent().remove(widget)
            logging.error("disabling widget: %s, reason: %s", name, ex)

    async def run_widgets(self):

        self._init_widgets()

        async with anyio.create_task_group() as grp:
            for widget in self.widget_list:
                name = widget.get_name()
                grp.start_soon(self._run_widget, widget, name, name=name)


class MehBar(Gtk.Application):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.win = None

    def do_activate(self, *args, **kwargs):
        if (active_window := self.get_active_window()) is not None:
            active_window.present()
        else:
            try:
                self.win = MehBarGUI(application=self)

                t_module_worker = Thread(
                    target=partial(anyio.run, self.win.run_widgets), name="AIO Worker"
                )
                t_module_worker.daemon = True
                t_module_worker.start()
                self.win.present()
            except Exception as ex:
                logging.critical("initialization failed: %s", ex)
                sys.exit(os.EX_SOFTWARE)


def entrypoint(argv):
    app = MehBar(
        application_id="org.codeberg.mehsayer.mehbar",
        flags=Gio.ApplicationFlags.FLAGS_NONE,
    )
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, app.quit)
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, app.quit)
    app.run(argv)
