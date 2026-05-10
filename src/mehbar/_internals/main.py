#!/usr/bin/env python3
# ruff: noqa: E402

from __future__ import annotations

import importlib
import logging
import os
import signal
import sys
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
    gi.require_version("GdkPixbuf", "2.0")
except ValueError as ex:
    logging.critical(str(ex))
    sys.exit(1)

from gi.repository import Gdk, Gio, GLib, Gtk, Gtk4LayerShell

import mehbar._widgets as builtin_widgets
from mehbar.exceptions import BarConfigError
from mehbar.widget import IconManager, WidgetBase

# GSK_RENDERER=cairo GDK_BACKEND=wayland


def get_config_home() -> Path:
    if (cfg_home := os.getenv("XDG_CONFIG_HOME")) is None:
        cfg_home = Path.home() / ".config"
    else:
        cfg_home = Path(cfg_home)
    return cfg_home / "mehbar"


def load_config() -> dict[str, Any]:  # noqa: MC0001
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
    # TODO: add yaml, jsonc

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
    WIDGETS = [
        "WidgetBacklight",
        "WidgetBattery",
        "WidgetBluetooth",
        "WidgetCPUPercentage",
        "WidgetDateTime",
        "WidgetDiskUsage",
        "WidgetExecRepeat",
        "WidgetExecTail",
        "WidgetFanSpeed",
        "WidgetFile",
        "WidgetI3KeyboardLayout",
        "WidgetI3Mode",
        "WidgetI3Scratchpad",
        "WidgetI3Window",
        "WidgetI3Workspaces",
        "WidgetMemoryUsage",
        "WidgetNetworkRate",
        "WidgetPlayerCtl",
        "WidgetPulseVolume",
        "WidgetSession",
        "WidgetStatic",
        "WidgetTemperature",
        "WidgetWifi",
        "WidgetWired",
    ]

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

        self.wtype_map = {}

        for widget_cls_name in self.WIDGETS:
            if (cl := getattr(builtin_widgets, widget_cls_name, None)) is not None:
                if (wtype := getattr(cl, "TYPE", None)) is None:
                    raise BarConfigError(f"unknown widget type for class '{cl!s}'")

                if wtype in self.wtype_map:
                    raise BarConfigError(f"duplicate widget type '{wtype}'")

                self.wtype_map[wtype] = cl

        self.i3_conn = None
        self._unique_wtypes = set()

        self.config = load_config()

        bar_config = self.config.get("bar")

        self.icon_manager = IconManager(bar_config.get("icon_size", 12))

        if (icons := self.config.get("icons")) is not None:
            for name, path in icons.items():
                self.icon_manager.load_image(name, path)

        if not bar_config or bar_config is None:
            raise RuntimeError("no valid configuration available")

        is_homogenous = bar_config.get("homogenous", self.DEFAULT_HOMOGENOUS)

        anchor = self.ANCHOR_MAP.get(
            bar_config.get("position", self.DEFAULT_ANCHOR), self.DEFAULT_ANCHOR
        )

        layer = self.LAYER_MAP.get(
            bar_config.get("layer", self.DEFAULT_LAYER), self.DEFAULT_LAYER
        )

        height = bar_config.get("height", self.MIN_HEIGHT)

        total_width = bar_config.get("width", 0)

        gaps = bar_config.get("gaps", self.DEFAULT_GAPS)

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

    def _widget_class_for_type(self, wtype: str) -> WidgetBase:

        if (widget_cls := self.wtype_map.get(wtype, None)) is None:
            types = ", ".join(self.wtype_map.keys())
            raise BarConfigError(f"widget type '{wtype}' is not one of: {types}")

        widget_unique = getattr(widget_cls, "UNIQUE", True)

        if widget_unique:
            if wtype in self._unique_wtypes:
                raise BarConfigError(f"windget of type '{wtype}' must be unique")
            self._unique_wtypes.add(wtype)

        return widget_cls

    def new_widget_for(self, name: str, **kwargs) -> WidgetBase:
        widget = None

        if (wtype := kwargs.pop("type", None)) is None:
            raise BarConfigError("widget type not specified")

        if name is None:
            raise BarConfigError(
                f"no configuration key 'name' for widget of type '{wtype}'"
            )
        if name in self.SECTION_NAMES:
            raise BarConfigError(f"name '{name}' is not allowed for widgets")

        widget_cls = self._widget_class_for_type(wtype)

        max_width = kwargs.pop("max_width", 0)

        width = kwargs.pop("width", 0)

        onclick = kwargs.pop("onclick", None)

        onscroll = kwargs.pop("onscroll", None)

        args = set()

        for name_, param in signature(widget_cls).parameters.items():
            args.add(name_)
            if name_ == "i3_conn":
                kwargs[name_] = self.i3_conn

            if name_ == "icon_manager":
                kwargs[name_] = self.icon_manager

            if param.default == Parameter.empty and name_ not in kwargs:
                raise BarConfigError(
                    f"no configuration key '{name_}' for widget '{name}'"
                )

        for del_arg in set(kwargs) - args:
            del kwargs[del_arg]
            logging.warning(
                "ignoring unknown configuration key '%s' for widget '%s'",
                del_arg,
                name,
            )

        widget = widget_cls(**kwargs)

        widget.set_name(name)

        if onclick is not None:
            for button, cmdline in enumerate(onclick):
                widget.onclick_exec(button, cmdline)

        if onscroll is not None:
            widget.onscroll_exec(*onscroll)

        if width > 0:
            widget.set_width_chars(width)

        if max_width > 0:
            widget.set_max_width_chars(max_width)

        return widget

    async def _run_widget(self, widget: Gtk.WidgetBase, name: str | None = None):
        try:
            await widget.run_wrapper()
        except Exception as ex:
            widget.shutdown()
            GLib.idle_add(widget.set_visible, False)
            parent = widget.get_parent()
            GLib.idle_add(parent.remove, widget)
            logging.error("disabling widget '%s': %s", name, ex)

    async def run_widgets(self):
        async with anyio.create_task_group() as grp:
            for section in self.SECTION_NAMES:
                box = getattr(self, f"{section}_box")

                if section in self.config:
                    for name, kwargs in self.config[section].items():
                        try:
                            widget = self.new_widget_for(name, **kwargs)
                        except BarConfigError as ex:
                            wtype = kwargs.get("type", "unknown")
                            logging.error(
                                "disabling widget '%s' of type '%s': %s",
                                name,
                                wtype,
                                ex,
                            )
                        else:
                            GLib.idle_add(box.append, widget)
                            grp.start_soon(self._run_widget, widget, name, name=name)
                else:
                    GLib.idle_add(box.set_visible, False)


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
