# ruff: noqa: E402
import logging
from typing import Any

import gi

gi.require_version("Playerctl", "2.0")

from gi.repository import GLib, Gtk, Playerctl  # type: ignore

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import BarWidget, WidgetBase, WidgetContent


def format_time(seconds: int) -> str:
    t_min, t_sec = divmod(max(seconds, 0), 60)
    return f"{t_min}:{t_sec:02d}"


class PlayerctlButton(WidgetBase):
    """Must be used from GTK main thread only. `alt_texts` are other texts
    the button may show, e.g. the pause label of the play button."""

    def __init__(
        self,
        name: str,
        res_mgr: ResourceManager,
        label_format: str | None = None,
        empty_text: str | None = None,
        alt_texts: tuple[str, ...] = (),
    ):
        super().__init__(
            name,
            res_mgr,
            {"label": label_format or ""},
            layout_specs=(empty_text or "", *alt_texts),
        )

        self.empty_text = empty_text or ""
        self.add_css_class("playerctl-button")
        self.reset()

    def show_text(self, text: str):
        self.apply_content(WidgetContent.parse(text))

    def show_fields(self, **fields: Any):
        self.apply_content(self.get_content(**fields))

    def reset(self):
        self.show_text(self.empty_text)


class WidgetPlayerCtl(BarWidget):
    """Media player controls. Unlike other widgets, it runs entirely on GTK
    main thread: Playerctl signals are delivered there."""

    MAX_SCROLL_SPEED = 100
    MIN_SCROLL_SPEED = 1
    DEFAULT_SEEK_OFFSET = 10
    TICKER_STEP = 10

    TYPE = "playerctl"
    STATIC = True

    DEFAULT_MODULES = [
        {"type": "previous", "label": "[icon=skip-back;]"},
        {"type": "seek_back", "label": "[icon=rewind;]"},
        {
            "type": "shuffle",
            "label_on": "[icon=shuffle;]",
            "label_off": "[icon=queue;]",
        },
        {
            "type": "play_pause",
            "label_play": "[icon=play;]",
            "label_pause": "[icon=pause;]",
        },
        {"type": "seek_forward", "label": "[icon=fast-forward;]"},
        {"type": "next", "label": "[icon=skip-forward;]"},
        {
            "type": "title",
            "label_empty": "[icon=music-note;]-----",
            "ticker": True,
            "scroll_speed": 10,
            "scroll_width": 128,
            "label_format": "[icon=music-note;]{artist} - {album} - {title}",
        },
        {
            "type": "time",
            "label_empty": "[icon=timer;]--:--",
            "label_format": "[icon=timer;]{current}/{total}",
        },
    ]

    MODULE_TYPES = frozenset(
        {
            "previous",
            "next",
            "seek_back",
            "seek_forward",
            "shuffle",
            "play_pause",
            "title",
            "time",
            "volume",
        }
    )

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        player_names = self.cfg.get("player_names") or []

        if not isinstance(player_names, list):
            raise BarConfigError("'player_names' must be a list")

        # any player if empty
        self.player_names = set(player_names)

        self.always_show = self.cfg.get("always_show", True)
        tick_ms = min(max(self.cfg.get("tick_ms", 500), 10), 2000)
        self.seek_offset = self.cfg.get("seek_offset", self.DEFAULT_SEEK_OFFSET)

        modules = self.cfg.get("modules") or self.DEFAULT_MODULES

        if not isinstance(modules, list):
            raise BarConfigError("'modules' must be a list")

        self.player: Playerctl.Player | None = None
        self.t_total = 0
        self._t_last = -1
        self.ticker = False
        self.ticker_direction = 0
        self.h_adj: Gtk.Adjustment | None = None

        self.btn_play_pause = None
        self.btn_shuffle = None
        self.btn_time = None
        self.btn_title = None
        self.btn_volume = None

        for module in modules:
            if (mod_type := module.get("type")) not in self.MODULE_TYPES:
                logging.warning("playerctl: unknown module type '%s'", mod_type)
                continue

            self._add_module(mod_type, module)

        self.add_css_class("playerctl")

        self.manager = Playerctl.PlayerManager()
        self.manager.connect("name-appeared", self._on_name_appeared)
        self.manager.connect("player-vanished", self._on_player_vanished)

        for player_name in self.manager.props.player_names:
            self._init_player(player_name)

        if self.player is None:
            self._on_status(None, Playerctl.PlaybackStatus.STOPPED)

        GLib.timeout_add(tick_ms, self._tick)

    def _button(
        self,
        name: str,
        label: str,
        action=None,
        *args,
        alt_texts: tuple[str, ...] = (),
    ) -> PlayerctlButton:
        button = PlayerctlButton(
            name, self.res_mgr, empty_text=label, alt_texts=alt_texts
        )
        if action is not None:
            button.onclick_call(1, self._player_call, action, *args)
        self.append(button)
        return button

    def _add_module(self, mod_type: str, module: dict[str, Any]):
        match mod_type:
            case "play_pause":
                self.label_play = module.get("label_play", "play")
                self.label_pause = module.get("label_pause", "pause")
                self.btn_play_pause = self._button(
                    "playerctl-play-pause",
                    self.label_play,
                    "play_pause",
                    alt_texts=(self.label_pause,),
                )
            case "next":
                self._button("playerctl-next", module.get("label", "next"), "next")
            case "previous":
                self._button(
                    "playerctl-previous", module.get("label", "prev"), "previous"
                )
            case "seek_back":
                self._button(
                    "playerctl-seek-back",
                    module.get("label", "rw"),
                    "seek",
                    -self.seek_offset * 10**6,
                )
            case "seek_forward":
                self._button(
                    "playerctl-seek-forward",
                    module.get("label", "ff"),
                    "seek",
                    self.seek_offset * 10**6,
                )
            case "shuffle":
                self.label_shuffle_on = module.get("label_on", "shuffle")
                self.label_shuffle_off = module.get("label_off", "no shuffle")
                self.btn_shuffle = self._button(
                    "playerctl-shuffle",
                    self.label_shuffle_off,
                    alt_texts=(self.label_shuffle_on,),
                )
                self.btn_shuffle.onclick_call(1, self._toggle_shuffle)
            case "time":
                self.btn_time = PlayerctlButton(
                    "playerctl-time",
                    self.res_mgr,
                    module.get("label_format", "{current}/{total}"),
                    module.get("label_empty", "--:--"),
                )
                self.append(self.btn_time)
            case "volume":
                self.btn_volume = PlayerctlButton(
                    "playerctl-volume",
                    self.res_mgr,
                    module.get("format", "{volume}%"),
                    module.get("label_empty"),
                )
                self.append(self.btn_volume)
            case "title":
                self._add_title(module)

    def _add_title(self, module: dict[str, Any]):
        scroll_speed = module.get("scroll_speed", 10)
        scroll_width = module.get("scroll_width", 0)

        self.ticker = module.get("ticker", False)
        self.btn_title = PlayerctlButton(
            "playerctl-title",
            self.res_mgr,
            module.get("label_format", "{artist} - {album} - {title}"),
            module.get("label_empty", ""),
        )

        scroll_view = Gtk.ScrolledWindow.new()

        if scroll_width > 0:
            scroll_speed = max(
                self.MIN_SCROLL_SPEED, min(scroll_speed, self.MAX_SCROLL_SPEED)
            )
            scroll_view.set_min_content_width(scroll_width)
            scroll_view.set_size_request(scroll_width, -1)
            scroll_view.set_policy(Gtk.PolicyType.EXTERNAL, Gtk.PolicyType.NEVER)

            viewport = Gtk.Viewport.new()
            viewport.set_child(self.btn_title)
            viewport.set_scroll_to_focus(False)

            h_adj = scroll_view.get_hadjustment()
            self.h_adj = h_adj

            def _scroll(_ctrl, _dx, dy):
                h_adj.set_value(h_adj.get_value() + (dy * scroll_speed))
                return True

            scroll_ctrl = Gtk.EventControllerScroll.new(
                Gtk.EventControllerScrollFlags.VERTICAL
            )
            scroll_ctrl.connect("scroll", _scroll)
            viewport.add_controller(scroll_ctrl)
            scroll_view.set_child(viewport)
        else:
            scroll_view.set_propagate_natural_width(True)
            scroll_view.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.NEVER)
            scroll_view.set_child(self.btn_title)

        self.append(scroll_view)

    # Players

    def _init_player(self, player_name: Playerctl.PlayerName):
        if self.player_names and player_name.name not in self.player_names:
            return

        player = Playerctl.Player.new_from_name(player_name)

        player.connect("playback-status", self._on_status)
        player.connect("metadata", self._on_metadata)
        player.connect("volume", self._on_volume)
        player.connect("seeked", self._on_seek)
        player.connect("shuffle", self._on_shuffle)
        self.manager.manage_player(player)

        props = player.props

        if not props.can_control:
            logging.warning("player %s cannot be controlled", player_name.name)

        self.player = player

        self._on_metadata(player, props.metadata)
        self._on_volume(player, props.volume)
        self._on_shuffle(player, props.shuffle)
        self._on_status(player, props.playback_status)

    def _on_name_appeared(self, _manager, player_name: Playerctl.PlayerName):
        self._init_player(player_name)

    def _on_player_vanished(self, manager, player: Playerctl.Player):
        if player is not self.player:
            return

        self.player = None

        if manager.props.players:
            # switch to another managed player
            self.player = manager.props.players[0]
            self._on_metadata(self.player, self.player.props.metadata)
            self._on_status(self.player, self.player.props.playback_status)
        else:
            self._on_status(None, Playerctl.PlaybackStatus.STOPPED)

    def _player_call(self, method: str, *args):
        if self.player is not None:
            try:
                getattr(self.player, method)(*args)
            except GLib.Error as ex:
                logging.error("cannot perform action %s: %s", method, ex.message)

    def _toggle_shuffle(self):
        if self.player is not None:
            self._player_call("set_shuffle", not self.player.props.shuffle)

    # Player signals

    def _on_metadata(self, player: Playerctl.Player, metadata: GLib.Variant | None):
        if player is not self.player:
            return

        meta = metadata.unpack() if metadata is not None else {}

        artists = meta.get("xesam:artist")
        artist = artists[0] if artists else "Unknown Artist"
        album = meta.get("xesam:album") or "Unknown Album"
        title = meta.get("xesam:title") or "Unknown Title"

        self.t_total = int(meta.get("mpris:length", 0) / 10**6)
        self._t_last = -1

        if self.btn_title is not None:
            self.btn_title.show_fields(artist=artist, album=album, title=title)

        if self.h_adj is not None:
            self.h_adj.set_value(0)

    def _on_volume(self, player: Playerctl.Player, volume: float):
        if player is self.player and self.btn_volume is not None:
            self.btn_volume.show_fields(volume=round(volume * 100))

    def _on_shuffle(self, player: Playerctl.Player, shuffle_status: bool):
        if player is self.player and self.btn_shuffle is not None:
            if shuffle_status:
                self.btn_shuffle.show_text(self.label_shuffle_on)
            else:
                self.btn_shuffle.show_text(self.label_shuffle_off)

    def _on_seek(self, player: Playerctl.Player, *_):
        if player is self.player:
            self._t_last = -1
            self._show_time()

    def _on_status(
        self, player: Playerctl.Player | None, status: Playerctl.PlaybackStatus
    ):
        if player is not None and player is not self.player:
            if status != Playerctl.PlaybackStatus.PLAYING:
                return
            # follow the player that started playing
            self.player = player
            self._on_metadata(player, player.props.metadata)

        self._t_last = -1

        if status == Playerctl.PlaybackStatus.PLAYING:
            self.set_visible(True)
            self._show_play_button(False)
            self._show_time()
        elif status == Playerctl.PlaybackStatus.PAUSED:
            self.set_visible(True)
            self._show_play_button(True)
            self._show_time()
        else:
            self.set_visible(self.always_show)
            self._show_play_button(True)
            self.t_total = 0

            for button in (self.btn_title, self.btn_time, self.btn_volume):
                if button is not None:
                    button.reset()

    def _show_play_button(self, is_play: bool):
        if self.btn_play_pause is not None:
            self.btn_play_pause.show_text(
                self.label_play if is_play else self.label_pause
            )

    def _show_time(self):
        if self.btn_time is None or self.player is None:
            return

        try:
            position = self.player.props.position // 10**6
        except GLib.Error:
            return

        if position != self._t_last:
            self._t_last = position
            self.btn_time.show_fields(
                current=format_time(position), total=format_time(self.t_total)
            )

    def _scroll_ticker(self):
        if not self.ticker or self.h_adj is None:
            return

        value = self.h_adj.get_value()

        if value + self.h_adj.get_page_size() >= self.h_adj.get_upper():
            self.ticker_direction = -1
        elif value <= 1:
            self.ticker_direction = 1

        self.h_adj.set_value(value + self.ticker_direction * self.TICKER_STEP)

    def _tick(self) -> bool:
        if (
            self.player is not None
            and self.player.props.playback_status == Playerctl.PlaybackStatus.PLAYING
        ):
            self._show_time()
            self._scroll_ticker()

        return GLib.SOURCE_CONTINUE
