#!/usr/bin/env python3
# ruff: noqa: E402

from __future__ import annotations

import ctypes
import logging
import os
import signal
import sys
from functools import partial
from pathlib import Path
from threading import Thread

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
        ctypes.CDLL(soname)
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

import mehbar._widgets as builtin_widgets

from .exceptions import BarConfigError
from .resource_manager import ResourceManager
from .widget import WidgetBase

# GSK_RENDERER=cairo GDK_BACKEND=wayland


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
        # "WidgetBacklight",
        # "WidgetBattery",
        # "WidgetBluetooth",
        "WidgetCPUUsage",
        "WidgetCPUFrequency",
        "WidgetDateTime",
        "WidgetDiskUsage",
        "WidgetApplication",
        "WidgetExecRepeat",
        "WidgetExecTail",
        # "WidgetFanSpeed",
        # "WidgetFile",
        "WidgetI3KeyboardLayout",
        # "WidgetI3Mode",
        # "WidgetI3Scratchpad",
        # "WidgetI3Window",
        # "WidgetI3Workspaces",
        "WidgetMemoryUsage",
        # "WidgetNetworkRate",
        # "WidgetPlayerCtl",
        "WidgetPulseVolume",
        # "WidgetSession",
        # "WidgetStatic",
        "WidgetTemperature",
        "WidgetWifi",
        # "WidgetWired",
    ]

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
    DEFAULT_ICON_SIZE = 16
    SECTION_NAMES = ["start", "center", "end"]

    def __init__(
        self,
        *args,
        cfg_dir: Path | None,
        theme: str | None,
        color_scheme: str,
        **kwargs: str,
    ):

        super().__init__(*args, **kwargs)

        self.wtype_map = {}

        for widget_cls_name in self.WIDGETS:
            if (cl := getattr(builtin_widgets, widget_cls_name, None)) is not None:
                if (wtype := getattr(cl, "TYPE", None)) is None:
                    raise BarConfigError(f"unknown widget type for class '{cl!s}'")

                if wtype in self.wtype_map:
                    raise BarConfigError(f"duplicate widget type '{wtype}'")

                self.wtype_map[wtype] = cl

        self._unique_wtypes = set()

        self.res_mgr = ResourceManager(cfg_dir, theme, color_scheme)

        bar_cfg = self.res_mgr.cfg.get("bar")

        is_homogenous = bar_cfg.get("homogenous", self.DEFAULT_HOMOGENOUS)

        anchor = self.ANCHOR_MAP.get(
            bar_cfg.get("position", self.DEFAULT_ANCHOR), self.DEFAULT_ANCHOR
        )

        layer = self.LAYER_MAP.get(
            bar_cfg.get("layer", self.DEFAULT_LAYER), self.DEFAULT_LAYER
        )

        height = bar_cfg.get("height", self.MIN_HEIGHT)

        total_width = bar_cfg.get("width", 0)

        gaps = bar_cfg.get("gaps", self.DEFAULT_GAPS)

        if len(gaps) == 2:
            gaps *= 2
        elif len(gaps) != 4 or not all([gap >= 0 for gap in gaps]):
            gaps = [0, 0, 0, 0]

        if total_width <= self.MIN_WIDTH:
            total_width = get_primary_mon_width()

        Gtk4LayerShell.init_for_window(self)
        Gtk4LayerShell.set_layer(self, layer)
        Gtk4LayerShell.set_anchor(self, anchor, True)
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.TOP, gaps[0])
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.LEFT, gaps[1])
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.BOTTOM, gaps[2])
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.RIGHT, gaps[3])
        Gtk4LayerShell.auto_exclusive_zone_enable(self)

        self.set_default_size(total_width - (gaps[1] + gaps[3]), height)

        style_provider = Gtk.CssProvider()
        css_stylesheet = self.res_mgr.css
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
        self.main_box.set_valign(Gtk.Align.FILL)
        self.main_box.append(self.start_box)
        self.main_box.append(self.center_box)
        self.main_box.append(self.end_box)

        self.main_box.add_css_class("bar-horizontal")

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

    def new_widget_for(self, name: str) -> WidgetBase:
        widget = None

        widget_cfg = self.res_mgr.get_cfg_for_name(name)

        if (widget_type := widget_cfg.get("type", None)) is None:
            raise BarConfigError("widget type not specified")

        widget_cls = self._widget_class_for_type(widget_type)
        widget = widget_cls(name, self.res_mgr)
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

                if section in self.res_mgr.cfg:
                    for name, widget_cfg in self.res_mgr.cfg[section].items():
                        try:
                            widget = self.new_widget_for(name)
                        except BarConfigError as ex:
                            widget_type = widget_cfg.get("type", "unknown")
                            logging.error(
                                "disabling widget '%s' of type '%s': %s",
                                name,
                                widget_type,
                                ex,
                            )
                        else:
                            GLib.idle_add(box.append, widget)
                            grp.start_soon(self._run_widget, widget, name, name=name)
                else:
                    GLib.idle_add(box.set_visible, False)


class MehBar(Gtk.Application):
    def __init__(
        self,
        cfg_dir: Path | None,
        theme: str | None = None,
        color_scheme: str | None = "system",
        **kwargs: str,
    ):
        super().__init__(**kwargs)
        self.cfg_dir = cfg_dir
        self.theme = theme
        self.color_scheme = color_scheme
        self.win = None

    def do_activate(self, *args, **kwargs):
        if (active_window := self.get_active_window()) is not None:
            active_window.present()
        else:
            try:
                self.win = MehBarGUI(
                    cfg_dir=self.cfg_dir,
                    theme=self.theme,
                    color_scheme=self.color_scheme,
                    application=self,
                )

                t_module_worker = Thread(
                    target=partial(anyio.run, self.win.run_widgets), name="AIO Worker"
                )
                t_module_worker.daemon = True
                t_module_worker.start()
                self.win.present()
            except Exception as ex:
                logging.critical("initialization failed: %s", ex)
                sys.exit(os.EX_SOFTWARE)


def entrypoint(*args, **kwargs):

    app = MehBar(
        application_id="org.codeberg.mehsayer.mehbar",
        flags=Gio.ApplicationFlags.FLAGS_NONE,
        **kwargs,
    )
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, app.quit)
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, app.quit)
    app.run()
