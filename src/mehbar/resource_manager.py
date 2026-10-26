__lazy_modules__ = ["importlib", "mehbar", "copy"]

import asyncio
import copy
import importlib
import logging
from collections.abc import Callable, Coroutine
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import anyio
import gi

from . import tools
from .exceptions import BarConfigError

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, Gio, GLib, Gtk  # type: ignore  # noqa: E402


def detect_color_scheme() -> str | None:
    """Returns the color scheme preferred by the desktop, 'dark' or 'light',
    or `None` if there is no known preference."""

    settings = Gtk.Settings.get_default()

    # GTK 4.20+ follows the XDG desktop portal setting
    if settings is not None and hasattr(Gtk, "InterfaceColorScheme"):
        match settings.props.gtk_interface_color_scheme:
            case Gtk.InterfaceColorScheme.DARK:
                return "dark"
            case Gtk.InterfaceColorScheme.LIGHT:
                return "light"

    schema_name = "org.gnome.desktop.interface"
    source = Gio.SettingsSchemaSource.get_default()

    if source is not None and (schema := source.lookup(schema_name, True)):
        if schema.has_key("color-scheme"):
            match Gio.Settings.new(schema_name).get_string("color-scheme"):
                case "prefer-dark":
                    return "dark"
                case "prefer-light":
                    return "light"

    if settings is not None and settings.props.gtk_application_prefer_dark_theme:
        return "dark"

    return None


class EventLoopBridge:
    """Schedules calls on the widget event loop from other threads, such as
    GTK main thread, without blocking the caller."""

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: set[asyncio.Task] = set()

    def attach(self):
        """Must be called from the event loop thread."""
        self._loop = asyncio.get_running_loop()

    def detach(self):
        self._loop = None

    def call_soon(self, func: Callable, *args: Any) -> bool:
        if (loop := self._loop) is None:
            return False
        try:
            loop.call_soon_threadsafe(func, *args)
        except RuntimeError:  # the loop is closed
            return False
        return True

    def run_soon(self, coro_func: Callable[..., Coroutine], *args: Any) -> bool:
        """Starts a fire-and-forget task, exceptions are logged."""

        def _done(task: asyncio.Task):
            self._tasks.discard(task)

            if not task.cancelled() and (ex := task.exception()) is not None:
                logging.error(
                    "%s failed: %s",
                    getattr(coro_func, "__name__", ""),
                    tools.describe_exception(ex),
                    extra={"exception": ex},
                )

        def _start():
            # the loop keeps only weak references to tasks
            task = asyncio.ensure_future(coro_func(*args))
            self._tasks.add(task)
            task.add_done_callback(_done)

        return self.call_soon(_start)


class _ConfigSource:
    """A directory or a GResource prefix that may contain configuration,
    style sheets and icons."""

    def __init__(self, label: str):
        self.label = label

    def read(self, name: str) -> bytes | None:
        raise NotImplementedError()

    def icon_file(self, path: str) -> Gio.File | None:
        raise NotImplementedError()

    def __str__(self) -> str:
        return self.label


class _DirSource(_ConfigSource):
    def __init__(self, path: Path):
        super().__init__(str(path))
        self.path = path

    def read(self, name: str) -> bytes | None:
        try:
            return (self.path / name).read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            return None

    def icon_file(self, path: str) -> Gio.File | None:
        if (fpath := self.path / "icons" / path).is_file():
            return Gio.File.new_for_path(str(fpath))
        return None


class _ResourceSource(_ConfigSource):
    def __init__(self, prefix: str):
        super().__init__("resource://" + prefix)
        self.prefix = prefix

    def read(self, name: str) -> bytes | None:
        try:
            data = Gio.resources_lookup_data(
                f"{self.prefix}/{name}", Gio.ResourceLookupFlags.NONE
            )
        except GLib.Error:
            return None
        return data.get_data()

    def exists(self) -> bool:
        try:
            Gio.resources_enumerate_children(self.prefix, Gio.ResourceLookupFlags.NONE)
        except GLib.Error:
            return False
        return True

    def icon_file(self, path: str) -> Gio.File | None:
        res_file = Gio.File.new_for_uri(f"resource://{self.prefix}/icons/{path}")
        return res_file if res_file.query_exists() else None


class ResourceManager:
    MIN_ICON_SIZE = 8
    DEFAULT_ICON_SIZE = 24
    COMMON_RESOURCE_PREFIX = "/org/mehbar/themes"
    BASE_THEME = "_base"
    BASE_RESOURCE_FILE = "_base.gresource"

    WIDGET_CFG_SECTIONS = ("start", "center", "end")
    COLOR_SCHEMES = ("light", "dark")
    DEFAULT_COLOR_SCHEME = "dark"

    # wake-ups of widgets are spread over time, see `relax_interval()`
    INTERVAL_OFFSET = 0.5
    INTERVAL_MAX_DEVIATION = 2.5
    # spreads phases of widgets sharing an interval evenly, whatever their number
    PHASE_STEP = (5**0.5 - 1) / 2

    # configuration file extensions and modules to parse them, in order of preference
    CONFIG_PARSERS = (
        ("toml", ("tomllib", "tomli")),
        ("json", ("json",)),
        ("jsonc", ("json5", "pyjson5")),
        ("json5", ("json5", "pyjson5")),
        ("yaml", ("yaml",)),
    )

    def __init__(
        self,
        cfg_dir: Path,
        theme: str | None,
        color_scheme: str | None,
    ):
        self.cfg_dir = cfg_dir
        self.loop = EventLoopBridge()

        self._icons: dict[str, Gdk.Paintable | None] = {}
        # relaxed interval -> number of widgets using it
        self._intervals: dict[float, int] = {}
        self._i3_connection = None
        self._i3_lock: anyio.Lock | None = None

        self.icon_theme = Gtk.IconTheme.get_for_display(Gdk.Display.get_default())

        self._load_resources()

        self._base_src = _ResourceSource(
            f"{self.COMMON_RESOURCE_PREFIX}/{self.BASE_THEME}"
        )
        self._user_src = _DirSource(cfg_dir)

        # The theme and the color scheme may be set in configuration files, so
        # read them without color scheme overlays first
        base_cfg = self._read_cfg(self._base_src, None)
        user_cfg = self._read_cfg(self._user_src, None)

        self._theme = self._resolve_theme(theme, base_cfg, user_cfg)
        self._theme_srcs = self._get_theme_sources(self._theme)

        if not self._theme_srcs:
            # not found
            self._theme = None

        theme_cfg: dict[str, Any] = {}
        for src in self._theme_srcs:
            tools.overlay_dict_r(theme_cfg, self._read_cfg(src, None))

        self._color_scheme = self._resolve_color_scheme(
            color_scheme, base_cfg, theme_cfg, user_cfg
        )

        logging.debug(
            "using theme '%s', color scheme '%s'", self.theme, self.color_scheme
        )

        layers = [
            self._read_cfg(src, self._color_scheme)
            for src in (self._base_src, *self._theme_srcs, self._user_src)
        ]

        self._cfg = self._merge_cfg_layers(layers[0], layers[1:])
        self._widget_cfgs = self._index_widget_cfgs(self._cfg)

        self._pixel_size = self._get_pixel_size()
        self._preload_icons()

    # Configuration

    @property
    def cfg(self) -> dict[str, Any]:
        return self._cfg

    @property
    def bar_cfg(self) -> dict[str, Any]:
        return self._cfg.get("bar", {})

    @property
    def theme(self) -> str:
        return self._theme or self.BASE_THEME

    @property
    def color_scheme(self) -> str:
        return self._color_scheme

    @property
    def pixel_size(self) -> int:
        return self._pixel_size

    def _get_pixel_size(self) -> int:
        try:
            pixel_size = int(self.bar_cfg.get("icon_size", self.DEFAULT_ICON_SIZE))
        except (TypeError, ValueError) as ex:
            raise BarConfigError(f"invalid icon size: {ex}") from ex
        return max(self.MIN_ICON_SIZE, pixel_size)

    def _load_resources(self):
        assets_dir = Path(str(resources.files("mehbar") / "assets"))
        base_res_path = assets_dir / self.BASE_RESOURCE_FILE

        if not base_res_path.is_file():
            raise RuntimeError(f"base resource file '{base_res_path}' not found")

        # the base theme, themes shipped with mehbar, then user themes
        res_paths = [base_res_path]
        res_paths.extend(
            path
            for path in sorted(assets_dir.glob("*.gresource"))
            if path != base_res_path
        )
        res_paths.extend(sorted((self.cfg_dir / "themes").glob("*.gresource")))

        for path in res_paths:
            try:
                Gio.Resource.load(str(path))._register()
            except GLib.Error as ex:
                logging.error("cannot load resource file '%s': %s", path, ex.message)

    def _resolve_theme(
        self, theme: str | None, base_cfg: dict[str, Any], user_cfg: dict[str, Any]
    ) -> str | None:
        if theme is None:
            theme = user_cfg.get("bar", {}).get("theme")
        if theme is None:
            theme = base_cfg.get("bar", {}).get("theme")
        if theme == self.BASE_THEME:
            theme = None
        return theme

    def _get_theme_sources(self, theme: str | None) -> list[_ConfigSource]:
        if theme is None:
            return []

        srcs: list[_ConfigSource] = []

        res_src = _ResourceSource(f"{self.COMMON_RESOURCE_PREFIX}/{theme}")
        if res_src.exists():
            srcs.append(res_src)

        if (theme_dir := self.cfg_dir / "themes" / theme).is_dir():
            srcs.append(_DirSource(theme_dir))

        if not srcs:
            logging.error("theme '%s' not found", theme)

        return srcs

    def _resolve_color_scheme(
        self, color_scheme: str | None, *cfgs: dict[str, Any]
    ) -> str:
        # 'system' on the command line overrides configuration files too
        if color_scheme is None:
            # the configuration with the highest precedence wins
            for cfg in reversed(cfgs):
                if (cs := cfg.get("bar", {}).get("color_scheme")) is not None:
                    color_scheme = cs
                    break

        if color_scheme not in self.COLOR_SCHEMES:
            if color_scheme not in (None, "system"):
                logging.warning("unknown color scheme '%s'", color_scheme)
            color_scheme = detect_color_scheme() or self.DEFAULT_COLOR_SCHEME

        return color_scheme

    def _parse_cfg(
        self, data: bytes, modules: tuple[str, ...], label: str
    ) -> dict[str, Any] | None:
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            parse = getattr(module, "safe_load", None) or module.loads

            try:
                cfg = parse(data.decode())
            except Exception as ex:
                raise BarConfigError(f"cannot parse '{label}': {ex}") from ex

            if cfg is None:
                cfg = {}
            elif not isinstance(cfg, dict):
                raise BarConfigError(f"'{label}': configuration must be a mapping")

            logging.debug("loaded configuration '%s' using '%s'", label, module_name)
            return cfg

        logging.warning(
            "cannot parse '%s', none of the modules is available: %s",
            label,
            ", ".join(modules),
        )
        return None

    def _read_cfg_file(self, src: _ConfigSource, stem: str) -> dict[str, Any] | None:
        for ext, modules in self.CONFIG_PARSERS:
            name = f"{stem}.{ext}"
            if (data := src.read(name)) is not None:
                return self._parse_cfg(data, modules, f"{src}/{name}")
        return None

    def _read_cfg(self, src: _ConfigSource, color_scheme: str | None) -> dict[str, Any]:
        cfg = self._read_cfg_file(src, "config") or {}

        if color_scheme is not None:
            if cs_cfg := self._read_cfg_file(src, f"config-{color_scheme}"):
                tools.overlay_dict_r(cfg, cs_cfg)

        return cfg

    def _merge_cfg_layers(
        self, base: dict[str, Any], layers: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Settings are merged in order: base, theme, user. Widgets defined in
        the base configuration are only used if neither the theme nor the user
        configuration define any."""

        def has_widgets(cfg: dict[str, Any]) -> bool:
            return any(cfg.get(sect) for sect in self.WIDGET_CFG_SECTIONS)

        merged = copy.deepcopy(base)

        if any(has_widgets(layer) for layer in layers):
            for sect in self.WIDGET_CFG_SECTIONS:
                merged.pop(sect, None)

        for layer in layers:
            tools.overlay_dict_r(merged, layer)

        return merged

    def _index_widget_cfgs(self, cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
        widget_cfgs = {}

        for sect in self.WIDGET_CFG_SECTIONS:
            if not isinstance(sect_cfg := cfg.get(sect, {}), dict):
                raise BarConfigError(f"section '{sect}' must be a mapping")

            for name, widget_cfg in sect_cfg.items():
                if not isinstance(widget_cfg, dict):
                    raise BarConfigError(f"widget '{name}' must be a mapping")
                if name in widget_cfgs:
                    raise BarConfigError(f"duplicate widget name '{name}'")
                widget_cfgs[name] = widget_cfg

        return widget_cfgs

    def get_cfg_for_name(self, name: str) -> dict[str, Any]:
        return self._widget_cfgs.get(name, {})

    def get_css(self) -> list[tuple[str, bytes]]:
        """Returns style sheets in order of increasing precedence."""
        css = []

        for src in (self._base_src, *self._theme_srcs, self._user_src):
            for name in ("style.css", f"style-{self.color_scheme}.css"):
                if (data := src.read(name)) is not None:
                    css.append((f"{src}/{name}", data))
        return css

    def relax_interval(self, interval: float) -> tuple[float, float]:
        """Spreads widget wake-ups over time, so that widgets polling at the
        same interval neither cause periodic CPU and I/O spikes nor wait on
        each other.

        The interval is nudged up to be at least `INTERVAL_OFFSET` seconds
        apart from intervals handed out before, by no more than the interval
        itself and `INTERVAL_MAX_DEVIATION`, e.g. 3 s may become up to 5.5 s,
        5 s up to 7.5 s. If there is no such interval, the least shared one
        within that range is reused. Returns the interval and the phase of
        wake-ups, as a fraction of the interval, which keeps widgets sharing
        an interval from waking up together."""

        limit = interval + min(interval, self.INTERVAL_MAX_DEVIATION)
        relaxed = interval

        while close := [
            i for i in self._intervals if abs(relaxed - i) < self.INTERVAL_OFFSET
        ]:
            relaxed = max(close) + self.INTERVAL_OFFSET

            if relaxed > limit:
                in_range = [i for i in self._intervals if interval <= i <= limit]
                relaxed = min(
                    in_range, key=lambda i: (self._intervals[i], i), default=interval
                )
                break

        shared = self._intervals.get(relaxed, 0)
        self._intervals[relaxed] = shared + 1

        return relaxed, shared * self.PHASE_STEP % 1.0

    # i3 / sway IPC

    async def get_i3_connection_async(self):
        """Returns the shared i3/sway IPC connection, connecting if there is
        none, and an event that is set once the connection is lost."""
        if self._i3_lock is None:
            self._i3_lock = anyio.Lock()

        async with self._i3_lock:
            if self._i3_connection is None:
                from ._widgets._i3 import I3Connection

                conn = await I3Connection().connect()
                self._i3_connection = (conn, anyio.Event())
                self.loop.run_soon(self._watch_i3_connection, conn)

        return self._i3_connection

    async def _watch_i3_connection(self, conn):
        try:
            # returns or raises once the connection is lost
            await conn.main()
            self.drop_i3_connection(conn)
        except Exception as ex:
            self.drop_i3_connection(conn, ex)
        finally:  # e.g. cancelled on exit
            self.drop_i3_connection(conn)

    def drop_i3_connection(self, conn, reason: BaseException | None = None):
        """Closes `conn`, if it is still the shared connection, and lets
        widgets using it know. Must be called from the event loop thread."""
        if self._i3_connection is None or self._i3_connection[0] is not conn:
            return

        _, lost = self._i3_connection
        self._i3_connection = None

        if reason is not None:
            logging.warning(
                "i3/sway IPC connection lost: %s", tools.describe_exception(reason)
            )

        conn.close()
        lost.set()

    # Icons

    def _preload_icons(self):
        icons = self._cfg.get("preload-icons") or {}

        if not isinstance(icons, dict):
            raise BarConfigError("section 'preload-icons' must be a mapping")

        for name, url in icons.items():
            self._icons[name] = self._load_icon(name, str(url))

    def _get_themed_icon(self, icon_name: str) -> Gdk.Paintable:
        if not self.icon_theme.has_icon(icon_name):
            logging.warning("icon '%s' not found in icon theme", icon_name)

        return self.icon_theme.lookup_icon(
            icon_name,
            None,
            self.pixel_size,
            1,
            Gtk.TextDirection.NONE,
            Gtk.IconLookupFlags.NONE,
        )

    def _get_file_icon(self, res_file: Gio.File) -> Gdk.Paintable:
        return Gtk.IconPaintable.new_for_file(res_file, self.pixel_size, 1)

    def _load_icon(self, name: str, url: str) -> Gdk.Paintable | None:
        url_ = urlsplit(url)
        path = url_.netloc + url_.path

        try:
            match url_.scheme:
                case "icontheme":
                    return self._get_themed_icon(path)
                case "theme":
                    for src in (*reversed(self._theme_srcs), self._base_src):
                        if (res_file := src.icon_file(path)) is not None:
                            return self._get_file_icon(res_file)
                    raise FileNotFoundError(f"icon '{path}' not found in theme")
                case "file":
                    res_file = Gio.File.new_for_uri(url)
                    if not res_file.query_exists():
                        raise FileNotFoundError(
                            f"file '{res_file.get_path()}' does not exist"
                        )
                    return self._get_file_icon(res_file)
                case "":
                    raise ValueError("no such preloaded icon")
                case _:
                    raise ValueError(f"unknown URL scheme '{url_.scheme}'")
        except (OSError, ValueError, GLib.Error) as ex:
            logging.error("failed to load icon '%s': %s", name, ex)

        return self.icon_theme.lookup_icon(
            "image-missing",
            None,
            self.pixel_size,
            1,
            Gtk.TextDirection.NONE,
            Gtk.IconLookupFlags.NONE,
        )

    def get_paintable(self, name: str) -> Gdk.Paintable | None:
        """Must be called from GTK main thread."""
        if name not in self._icons:
            self._icons[name] = self._load_icon(name, name)
        return self._icons[name]
