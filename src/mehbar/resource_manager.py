__lazy_modules__ = ["importlib", "io", "mehbar", "copy"]

import copy
import importlib
import io
import itertools
import logging
import re

# import sys
from functools import lru_cache
from pathlib import Path
from typing import IO, Any
from urllib.parse import urlsplit

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, Gio, GLib, Gtk

from . import exceptions, tools

_INFINITY = float("inf")


class ResourceManager:
    MIN_ICON_SIZE = 8
    DEFAULT_ICON_SIZE = 24
    COMMON_RESOURCE_PREFIX = "/org/mehbar/themes"
    BASE_RESOURCE_PREFIX = "/org/mehbar/themes/_base"

    WIDGET_CFG_SECTIONS = ["start", "center", "end"]
    BAR_CFG_SECTIONS = ["bar", "preload-icons"]

    MAX_WIDGETS = 32

    INTERVAL_OFFSET = 0.5

    CONFIG_PARSER_MAP = {
        "tomli": "toml",
        "toml": "toml",
        "tomllib": "toml",
        "json5": "json5",
        "pyjson5": "json5",
        "jsonc": "jsonc",
        "json": "json",
        "yaml": "yaml",
    }

    def __init__(
        self,
        cfg_dir: Path,
        theme: str | None,
        color_scheme: str | None,
    ):

        self._icons = {}
        self._cksums = {}
        self.cfg_dir = cfg_dir

        self._theme = theme
        self._color_scheme = color_scheme
        self._intervals = set()

        self._i3_connection = None
        self._pixel_size = 0

        self._css = b""
        self._cfg = {}

        self._widget_cfg = {}

        self.icon_theme = Gtk.IconTheme.get_for_display(Gdk.Display.get_default())

        self._name_re = re.compile(r"^[a-zA-Z0-9-_]{3,16}$")

        self._get_themed_icon = lru_cache(maxsize=128)(self._get_themed_icon)
        self._get_resource_icon = lru_cache(maxsize=128)(self._get_resource_icon)
        self.get_cfg_for_name = lru_cache(maxsize=32)(self.get_cfg_for_name)

        self._load_resources()
        self._preload_icons()

    @staticmethod
    def __readonly():
        raise AttributeError("attribute is read-only")

    def _get_relaxed_interval(self, interval: int | float) -> int | float:

        # makes sure that all intervals are at least 500 milliseconds apart from each other
        adjusted_interval = interval + self.INTERVAL_OFFSET

        # make sure that there is at least one pair of numbers (zero and infinity)
        for lo, hi in itertools.pairwise(sorted([0, *self._intervals, _INFINITY])):
            lo_threshold = lo + self.INTERVAL_OFFSET

            if lo_threshold < interval < hi:
                break
            elif lo < adjusted_interval < hi:
                interval += self.INTERVAL_OFFSET - (interval - lo)
                break
            elif hi == _INFINITY and lo_threshold > interval:
                interval = lo_threshold
                break

        self._intervals.add(interval)

        return interval

    def _preload_icons(self):
        if (icons := self.cfg.get("preload-icons")) is not None:
            for name, path in icons.items():
                self._load_icon(name, path)

    def _load_resources(self):

        base_res_name = "_base.gresource"

        if (base_res_path := self.get_asset(base_res_name)) is None:
            raise RuntimeError(f"base resource file '{base_res_name}' not found")

        res_paths = [base_res_path]

        for res_path in (self.cfg_dir / "themes").glob("*.gresource"):
            res_paths.append(res_path)

        for path in res_paths:
            res = Gio.Resource.load(str(path))
            res._register()

    async def get_i3_connection_async(self):
        if self._i3_connection is None:
            from i3ipc.aio import Connection

            self._i3_connection = await Connection().connect()
        return self._i3_connection

    @property
    def pixel_size(self) -> int:

        if self._pixel_size == 0:
            pixel_size = self.cfg["bar"].get("icon_size", self.DEFAULT_ICON_SIZE)

            self._pixel_size = max(self.MIN_ICON_SIZE, pixel_size)
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, _: Any):
        self.__readonly()

    @property
    def cfg(self) -> tools.SelectorDict[str, Any]:

        if not self._cfg:
            self._cfg = self._read_cfg_dir(self.cfg_dir)

            if "bar" not in self._cfg:
                cfg_res = self._read_cfg_resource(self.BASE_RESOURCE_PREFIX)

                tools.overlay_dict_r(self._cfg, cfg_res)

            if (theme := self._cfg["bar"].get("theme")) is not None:
                path_theme_res = f"{self.COMMON_RESOURCE_PREFIX}/{theme}"
                cfg_theme = self._read_cfg_resource(path_theme_res)

                if not cfg_theme:
                    path_theme_dir = self.cfg_dir / "themes" / theme
                    cfg_theme = self._read_cfg_dir(path_theme_dir)
                tools.overlay_dict_r(self._cfg, cfg_theme)

        return tools.SelectorDict(self._cfg)

    @cfg.setter
    def cfg(self, _: Any):
        self.__readonly()

    @property
    def theme(self) -> str:
        if self._theme is None:
            if (bar_cfg := self.cfg.get("bar")) is not None:
                self._theme = bar_cfg.get("theme")
        return self._theme

    @theme.setter
    def theme(self, _: Any):
        self.__readonly()

    @property
    def color_scheme(self) -> str:
        if self._color_scheme is None:
            color_scheme_ = None
            if (bar_cfg := self._cfg.get("bar")) is not None:
                color_scheme_ = bar_cfg.get("color_scheme")

            if color_scheme_ is None:
                try:
                    from gi.repository import Gio

                    gsettings_schema = "org.gnome.desktop.interface"
                    source = Gio.SettingsSchemaSource.get_default()

                    if source.lookup(gsettings_schema) is not None:
                        gsettings = Gio.Settings.new(gsettings_schema)

                        gsettings_cs = gsettings.get_string("color-scheme")
                        if gsettings_cs == "prefer-dark":
                            color_scheme_ = "dark"
                        elif gsettings_cs == "prefer-light":
                            color_scheme_ = "light"
                except ImportError:
                    pass

            if color_scheme_ is None:
                try:
                    import gi

                    gi.require_version("Adw", "1")
                    from gi.repository import Adw

                    style_mgr = Adw.StyleManager.get_default()
                    color_scheme_ = "dark" if style_mgr.get_dark() else "light"
                except (ImportError, ValueError):
                    pass

                if color_scheme_ is not None:
                    self._color_scheme = color_scheme_
                else:
                    self._color_scheme = "system"

        return self._color_scheme

    @color_scheme.setter
    def color_scheme(self, _: Any):
        self.__readonly()

    @property
    def css(self):
        if not self._css:
            self._css = self._read_css_resource(self.BASE_RESOURCE_PREFIX)
            self._css += self._read_css_dir(self.cfg_dir)
            if self.theme is not None:
                theme_res_path = f"{self.COMMON_RESOURCE_PREFIX}/{self.theme}"
                self._css += self._read_css_resource(theme_res_path)
                theme_dir_path = self.cfg_dir / "themes" / self.theme
                self._css += self._read_css_dir(theme_dir_path)
        return self._css

    @css.setter
    def css(self, _: Any):
        self.__readonly()

    @staticmethod
    def get_asset(asset: str | Path) -> Path | None:

        from importlib import resources

        asset_file = None
        with resources.as_file(resources.files("mehbar")) as mod_dir:
            asset_file_ = mod_dir / "assets" / asset

            if asset_file_.is_file():
                asset_file = asset_file_
        return asset_file

    def get_cfg_for_name(self, name: str) -> tools.SelectorDict[str, Any]:
        cfg = tools.SelectorDict()

        for sect in self.WIDGET_CFG_SECTIONS:
            cfg_ = self.cfg.select(f"{sect}.{name}", None)
            if cfg_ is not None and isinstance(cfg_, dict):
                cfg = tools.SelectorDict(cfg_)

                interval = cfg.get("interval", 0)

                if interval > 0 and cfg.get("relax_interval", True):
                    cfg["interval"] = self._get_relaxed_interval(interval)

                break
        return cfg

    def _parse_cfg_io(self, fhandle: IO, parser) -> dict[str, Any]:

        cfg = {}

        try:
            parser = importlib.import_module(parser)
            parser_func = getattr(parser, "safe_load", parser.load)
            cfg = parser_func(fhandle)
        except Exception:
            logging.debug("cannot parse configuration file with '%s'", parser.__name__)
            pass

        return cfg

    def _parse_cfg_file(self, path: Path, parser: str) -> dict[str, Any]:

        parsed = {}

        with open(path, "rb") as fhandle:
            parsed = self._parse_cfg_io(fhandle, parser)

        return parsed

    def _parse_cfg_resource(self, path: str, parser_mod: str) -> dict[str, Any]:
        cfg = {}

        try:
            if res_bytes := Gio.resources_lookup_data(path, 0):
                with io.BytesIO(res_bytes.get_data()) as fhandle:
                    cfg = self._parse_cfg_io(fhandle, parser_mod)

            # res_bytes.unref()
        except GLib.GError:
            logging.debug("resource at '%s' does not exist", path)

        return cfg

    def _read_cfg_dir(self, cfg_dir: Path) -> dict[str, Any]:
        cfg = {}
        if cfg_dir.is_dir():
            for parser_mod, ext in self.CONFIG_PARSER_MAP.items():
                path_ = cfg_dir / f"config.{ext}"
                if path_.is_file():
                    cfg = self._parse_cfg_file(path_, parser_mod)

                if self._color_scheme is None and "bar" in cfg:
                    color_scheme = cfg["bar"].get("color_scheme")
                else:
                    color_scheme = self._color_scheme

                if color_scheme is not None:
                    path_ = cfg_dir / f"config-{color_scheme}.{ext}"
                    if path_.is_file():
                        cs_cfg = self._parse_cfg_file(path_, parser_mod)
                        tools.overlay_dict_r(cfg, cs_cfg)
                if cfg:
                    logging.debug(
                        "loaded configuration from '%s' using '%s'", cfg_dir, parser_mod
                    )
                    break
        return cfg

    def _read_cfg_resource(self, path: str) -> dict[str, Any]:

        cfg = {}
        for parser_mod, ext in self.CONFIG_PARSER_MAP.items():
            path_ = f"{path}/config.{ext}"

            cfg = self._parse_cfg_resource(path_, parser_mod)

            if self.color_scheme is None and "bar" in cfg:
                color_scheme = cfg["bar"].get("color_scheme")
            else:
                color_scheme = self.color_scheme

            if color_scheme is not None:
                path_ = f"{path}/config-{color_scheme}.{ext}"
                cfg_cs = self._parse_cfg_resource(path_, parser_mod)
                tools.overlay_dict_r(cfg, cfg_cs)

            if cfg:
                logging.debug(
                    "loaded configuration from '%s' using '%s'", path, parser_mod
                )
                break

        return cfg

    def _read_css_file(self, path: Path | str) -> bytes:
        css = b""

        with open(path, "rb") as fhandle:
            css += fhandle.read()
            css += b"\n"

        return css

    def _read_css_dir(self, css_dir: Path) -> bytes:
        css = b""

        try:
            css += self._read_css_file(css_dir / "style.css")
            if self.color_scheme is not None:
                css += self._read_css_file(css_dir / f"style-{self.color_scheme}.css")
        except Exception:
            pass

        return css

    def _read_css_resource_path(self, path: str) -> bytes:
        css = b""

        try:
            if res_bytes := Gio.resources_lookup_data(path, 0):
                css += res_bytes.get_data()
                css += b"\n"
            # res_bytes.unref()
        except GLib.GError:
            logging.debug("resource at '%s' does not exist", path)

        return css

    def _read_css_resource(self, path: str) -> bytes:

        path_ = f"{path}/style.css"
        css = self._read_css_resource_path(path_)

        if self.color_scheme is not None:
            path_cs = f"{path}/style-{self.color_scheme}.css"
            css += self._read_css_resource_path(path_cs)

        return css

    def _get_themed_icon(self, icon_id: str) -> Gdk.Paintable | None:
        paintable = None

        paintable = self.icon_theme.lookup_icon(
            icon_id,
            None,
            self.cfg["bar"]["icon_size"],
            1,
            Gtk.TextDirection.NONE,
            Gtk.IconLookupFlags.NONE,
        )

        if paintable:
            paintable = paintable.get_current_image()

        return paintable

    def _get_resource_icon(self, path: Path | str) -> Gdk.Paintable | None:

        paintable = None
        res_file = Gio.File.new_for_uri(str(path))

        try:
            paintable = Gtk.IconPaintable.new_for_file(
                res_file, self.cfg["bar"]["icon_size"], 1
            )
        except Exception:
            paintable = self._get_themed_icon("image-missing")

        return paintable

    def _load_icon(self, name: str, url: str):

        url_ = urlsplit(url)

        path = Path(url_.netloc, url_.path)

        paintable = None

        if url_.scheme == "icontheme":
            paintable = self._get_themed_icon(str(path))
        else:
            try:
                if url_.scheme == "theme":
                    if self.theme is None:
                        theme_ = "_base"
                    else:
                        theme_ = self.theme

                    res_path = f"{self.COMMON_RESOURCE_PREFIX}/{theme_}/icons/{path}"
                    paintable = self._get_resource_icon("resource://" + res_path)
                elif url_.scheme == "file":
                    cksum = tools.md5sum_sync(url_.path)

                    if cksum in self._cksums and self._cksums[cksum] in self.icons:
                        self._icons[name] = self._icons[self._cksums[cksum]]
                    else:
                        paintable = self._get_resource_icon(url)
                        self._cksums[cksum] = name
                else:
                    if url_.scheme:
                        logging.error(
                            "unknown URL scheme for icon '%s'",
                            name,
                        )
                    else:
                        logging.error("icon '%s' has not been preloaded", name)

            except Exception:
                logging.error("failed to load icon '%s' from '%s'", name, path)
                paintable = self._get_themed_icon("image-missing")

        self._icons[name] = paintable

    def get_paintable(self, name: str) -> Gdk.Paintable:
        if name not in self._icons:
            self._load_icon(name, name)
        return self._icons[name]
