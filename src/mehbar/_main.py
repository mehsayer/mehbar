#!/usr/bin/env python3
# ruff: noqa: E402

from __future__ import annotations

import ctypes
import logging
import os
import signal
import sys
import threading
from pathlib import Path

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

# GTK4 layer shell library must be loaded before libwayland-client, that is
# before GTK is imported
cdll_failed = []

for soname in ["libgtk4-layer-shell.so.0", "libgtk4-layer-shell.so.1"]:
    try:
        ctypes.CDLL(soname)
        cdll_failed.clear()
        break
    except OSError as ex:
        cdll_failed.append((soname, ex))

if cdll_failed:
    logging.critical("failed to load GTK4 layer shell library")
    for soname, ex in cdll_failed:
        logging.critical("tried '%s': %s", soname, ex)
    sys.exit(1)
try:
    gi.require_version("Gtk", "4.0")
    gi.require_version("Gdk", "4.0")
    gi.require_version("Gtk4LayerShell", "1.0")
except ValueError as ex:
    logging.critical(str(ex))
    sys.exit(1)

from gi.repository import Gdk, Gio, GLib, Gtk, Gtk4LayerShell  # type: ignore

from . import _widgets
from .exceptions import BarConfigError
from .resource_manager import ResourceManager
from .widget import BarWidget

# GSK_RENDERER=cairo GDK_BACKEND=wayland


def describe_exception(ex: BaseException) -> str:
    # report the first actual error, not the group wrapping it
    while isinstance(ex, BaseExceptionGroup) and ex.exceptions:
        ex = ex.exceptions[0]

    if isinstance(ex, BarConfigError):
        return str(ex)

    if msg := str(ex):
        return f"{type(ex).__name__}: {msg}"

    return type(ex).__name__


class MehBarGUI(Gtk.ApplicationWindow):
    ANCHOR_MAP = {"top": Gtk4LayerShell.Edge.TOP, "bottom": Gtk4LayerShell.Edge.BOTTOM}
    LAYER_MAP = {
        "top": Gtk4LayerShell.Layer.TOP,
        "bottom": Gtk4LayerShell.Layer.BOTTOM,
    }
    MIN_HEIGHT = 24
    MIN_WIDTH = 256
    DEFAULT_ANCHOR = "top"
    DEFAULT_LAYER = "top"
    DEFAULT_HOMOGENOUS = True
    SECTION_NAMES = ResourceManager.WIDGET_CFG_SECTIONS
    NAMESPACE = "mehbar"
    STOP_TIMEOUT = 3

    def __init__(
        self,
        *args,
        cfg_dir: Path,
        theme: str | None,
        color_scheme: str | None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.res_mgr = ResourceManager(cfg_dir, theme, color_scheme)

        self._worker: threading.Thread | None = None
        self._cancel_scope: anyio.CancelScope | None = None

        bar_cfg = self.res_mgr.bar_cfg

        is_homogenous = bool(bar_cfg.get("homogenous", self.DEFAULT_HOMOGENOUS))

        self._setup_layer_shell(bar_cfg)
        self._load_css()

        self.main_box = Gtk.Box.new(Gtk.Orientation.HORIZONTAL, 0)
        self.main_box.set_homogeneous(is_homogenous)
        self.main_box.set_valign(Gtk.Align.FILL)
        self.main_box.add_css_class("bar-horizontal")

        self.boxes: dict[str, Gtk.Box] = {}

        for section in self.SECTION_NAMES:
            box = Gtk.Box.new(Gtk.Orientation.HORIZONTAL, 0)
            box.set_name(section)
            box.set_valign(Gtk.Align.CENTER)
            self.boxes[section] = box
            self.main_box.append(box)

        self.boxes["start"].set_halign(Gtk.Align.START)
        self.boxes["end"].set_halign(Gtk.Align.END)

        if is_homogenous:
            self.boxes["center"].set_halign(Gtk.Align.CENTER)
        else:
            self.boxes["center"].set_halign(Gtk.Align.START)
            self.boxes["center"].set_hexpand(True)

        self.set_child(self.main_box)

        self.widgets = self._create_widgets()

    def _setup_layer_shell(self, bar_cfg: dict):
        position = bar_cfg.get("position", self.DEFAULT_ANCHOR)
        if (anchor := self.ANCHOR_MAP.get(position)) is None:
            logging.warning("unknown bar position '%s'", position)
            anchor = self.ANCHOR_MAP[self.DEFAULT_ANCHOR]

        layer_name = bar_cfg.get("layer", self.DEFAULT_LAYER)
        if (layer := self.LAYER_MAP.get(layer_name)) is None:
            logging.warning("unknown bar layer '%s'", layer_name)
            layer = self.LAYER_MAP[self.DEFAULT_LAYER]

        height = bar_cfg.get("height", self.MIN_HEIGHT)
        if not isinstance(height, int) or height < 1:
            raise BarConfigError("bar height must be a positive integer")

        width = bar_cfg.get("width", 0)
        if not isinstance(width, int):
            raise BarConfigError("bar width must be an integer")

        gaps = bar_cfg.get("gaps", [0, 0, 0, 0])

        if not isinstance(gaps, list) or not all(
            isinstance(gap, int) and gap >= 0 for gap in gaps
        ):
            logging.warning("bar gaps must be a list of non-negative integers")
            gaps = [0, 0, 0, 0]
        elif len(gaps) == 2:
            # vertical, horizontal
            gaps = gaps * 2
        elif len(gaps) != 4:
            logging.warning("bar gaps must be a list of either 2 or 4 integers")
            gaps = [0, 0, 0, 0]

        Gtk4LayerShell.init_for_window(self)
        Gtk4LayerShell.set_namespace(self, self.NAMESPACE)
        Gtk4LayerShell.set_layer(self, layer)
        Gtk4LayerShell.set_anchor(self, anchor, True)
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.TOP, gaps[0])
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.LEFT, gaps[1])
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.BOTTOM, gaps[2])
        Gtk4LayerShell.set_margin(self, Gtk4LayerShell.Edge.RIGHT, gaps[3])
        Gtk4LayerShell.auto_exclusive_zone_enable(self)

        if width > self.MIN_WIDTH:
            # fixed width, centered horizontally
            self.set_default_size(width, height)
        else:
            # stretch over the whole output
            Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.LEFT, True)
            Gtk4LayerShell.set_anchor(self, Gtk4LayerShell.Edge.RIGHT, True)
            self.set_default_size(-1, height)

    def _load_css(self):
        display = Gdk.Display.get_default()

        # later style sheets take precedence
        for prio, (label, css) in enumerate(self.res_mgr.get_css()):

            def _on_error(_provider, section, error, label=label):
                loc = section.get_start_location()
                logging.warning(
                    "%s:%d:%d: %s", label, loc.lines + 1, loc.line_chars + 1, error.message
                )

            provider = Gtk.CssProvider()
            provider.connect("parsing-error", _on_error)
            provider.load_from_bytes(GLib.Bytes.new(css))

            Gtk.StyleContext.add_provider_for_display(
                display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION + prio
            )

    def _new_widget(self, name: str, unique_types: set[str]) -> BarWidget:
        widget_cfg = self.res_mgr.get_cfg_for_name(name)

        if (wtype := widget_cfg.get("type")) is None:
            raise BarConfigError("widget type not specified")

        widget_cls = _widgets.get_widget_class(wtype)

        if widget_cls.UNIQUE:
            if wtype in unique_types:
                raise BarConfigError(f"widget of type '{wtype}' must be unique")
            unique_types.add(wtype)

        return widget_cls(name, self.res_mgr)

    def _create_widgets(self) -> list[tuple[str, BarWidget]]:
        widgets = []
        unique_types: set[str] = set()

        for section in self.SECTION_NAMES:
            box = self.boxes[section]

            for name, widget_cfg in self.res_mgr.cfg.get(section, {}).items():
                try:
                    widget = self._new_widget(name, unique_types)
                except Exception as ex:
                    logging.error(
                        "disabling widget '%s' of type '%s': %s",
                        name,
                        widget_cfg.get("type", "unknown"),
                        describe_exception(ex),
                    )
                else:
                    box.append(widget)
                    widgets.append((name, widget))

        return widgets

    # Widget event loop, runs on its own thread

    def start_widgets(self):
        self._worker = threading.Thread(
            target=self._worker_main, name="AIO Worker", daemon=True
        )
        self._worker.start()

    def stop_widgets(self):
        """Cancels widget tasks and waits for the event loop to finish."""
        if self._worker is None or not self._worker.is_alive():
            return

        if (scope := self._cancel_scope) is not None:
            self.res_mgr.loop.call_soon(scope.cancel)

        self._worker.join(self.STOP_TIMEOUT)

        if self._worker.is_alive():
            logging.warning("widgets did not stop in %d seconds", self.STOP_TIMEOUT)

    def _worker_main(self):
        try:
            anyio.run(self._run_widgets, backend="asyncio")
        except Exception as ex:
            logging.critical("widget event loop failed: %s", describe_exception(ex))

    async def _run_widgets(self):
        self.res_mgr.loop.attach()

        try:
            async with anyio.create_task_group() as grp:
                self._cancel_scope = grp.cancel_scope

                for name, widget in self.widgets:
                    grp.start_soon(self._run_widget, widget, name, name=name)

                # keep event driven widgets alive
                await anyio.sleep_forever()
        finally:
            self.res_mgr.loop.detach()

    def _remove_widget(self, widget: BarWidget) -> bool:
        if (parent := widget.get_parent()) is not None:
            parent.remove(widget)
        return GLib.SOURCE_REMOVE

    async def _run_widget(self, widget: BarWidget, name: str):
        try:
            await widget.run_wrapper()
        except Exception as ex:
            widget.shutdown()
            GLib.idle_add(self._remove_widget, widget)
            logging.error("disabling widget '%s': %s", name, describe_exception(ex))


class MehBar(Gtk.Application):
    def __init__(
        self,
        cfg_dir: Path,
        theme: str | None = None,
        color_scheme: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cfg_dir = cfg_dir
        self.theme = theme
        self.color_scheme = color_scheme
        self.win: MehBarGUI | None = None
        self.exit_status = os.EX_OK

    def do_activate(self):
        if self.win is not None:
            self.win.present()
            return

        try:
            self.win = MehBarGUI(
                cfg_dir=self.cfg_dir,
                theme=self.theme,
                color_scheme=self.color_scheme,
                application=self,
            )
        except BarConfigError as ex:
            logging.critical("configuration error: %s", ex)
            self.exit_status = os.EX_CONFIG
            self.quit()
            return
        except Exception as ex:
            logging.critical("initialization failed: %s", describe_exception(ex))
            self.exit_status = os.EX_SOFTWARE
            self.quit()
            return

        self.win.present()
        self.win.start_widgets()

    def do_shutdown(self):
        if self.win is not None:
            self.win.stop_widgets()
        Gtk.Application.do_shutdown(self)


def entrypoint(**kwargs) -> int:
    app = MehBar(
        application_id="org.codeberg.mehsayer.mehbar",
        flags=Gio.ApplicationFlags.DEFAULT_FLAGS,
        **kwargs,
    )

    def _quit():
        app.quit()
        return GLib.SOURCE_REMOVE

    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, _quit)
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, _quit)

    status = app.run()
    return status or app.exit_status
