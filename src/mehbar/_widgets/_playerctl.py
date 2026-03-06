from typing import Any
import asyncio
import logging

from gi.repository import GLib, Gtk
from gi.repository import Playerctl

from mehbar.tools import OptionalFormatter
from mehbar.widgets import BarWidget


class PlayerctlButton(BarWidget):
    def __init__(self, name: str, label: str, label_format: str | None = None):
        super().__init__(0, label_format)

        self.initial_label = label
        self.set_name(name)
        self.set_label(label)
        self.add_css_class("playerctl-button")

    def reset_label_idle(self):
        self.set_label_idle(self.initial_label)

    def update(self):
        raise NotImplementedError()


class BarWidgetPlayerCtl(Gtk.Box):

    MAX_SCROLL_SPEED = 100
    MIN_SCROLL_SPEED = 1

    SUPPORTED_MODULES = [
        "next",
        "play_pause",
        "previous",
        "seek_back",
        "seek_forward",
        "title",
        "volume",
        "time",
        "shuffle",
    ]

    def __init__(
        self,
        modules: list[dict[str, Any]],
        player_names: list[str],
        always_show: bool = False,
        tick_ms: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.log = logging.getLogger(self.__class__.__name__)

        if player_names is None:
            raise BarConfigError("no player names speficied")

        self.player_names = player_names

        self.set_name("playerctl")

        self.ticker_direction = 0

        self.t_total = 0

        self.formatter = OptionalFormatter()

        self.manager = Playerctl.PlayerManager()

        self.player = None

        self.player_ready = asyncio.Event()

        self._t_last = 0.0

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
        self.label_shuffle_on = None
        self.label_shuffle_off = None
        self.scroll_view = None
        self.vol_labels = []
        self.ticker = False
        self.btn_shuffle = None
        scroll_speed = self.MIN_SCROLL_SPEED

        for module in modules:

            mod_type = module.get("type")
            if mod_type not in self.SUPPORTED_MODULES:
                continue

            match mod_type:
                case "play_pause":
                    if "label_play" in module and "label_pause" in module:
                        self.label_play = module["label_play"]
                        self.label_pause = module["label_pause"]
                        self.btn_play_pause = PlayerctlButton(
                            "playerctl-play-pause", self.label_play
                        )
                        self.append(self.btn_play_pause)
                case "next":
                    if "label" in module:
                        self.btn_next = PlayerctlButton(
                            "playerctl-next", module["label"]
                        )
                        self.append(self.btn_next)
                case "previous":
                    if "label" in module:
                        self.btn_previous = PlayerctlButton(
                            "playerctl-previuos", module["label"]
                        )
                        self.append(self.btn_previous)
                case "seek_back":
                    if "label" in module:
                        self.btn_seek_back = PlayerctlButton(
                            "playerctl-seek-back", module["label"]
                        )
                        self.append(self.btn_seek_back)
                case "seek_forward":
                    if "label" in module:
                        self.btn_seek_forward = PlayerctlButton(
                            "playerctl-seek-forward", module["label"]
                        )
                        self.append(self.btn_seek_forward)
                case "shuffle":
                    if "label_on" in module and "label_off" in module:
                        self.label_shuffle_on = module["label_on"]
                        self.label_shuffle_off = module["label_off"]
                        self.btn_shuffle = PlayerctlButton(
                            "playerctl-shuffle", self.label_shuffle_off
                        )
                        self.append(self.btn_shuffle)
                case "time":
                    self.btn_time = PlayerctlButton(
                        "playerctl-time",
                        module.get("label_empty"),
                        module.get("label_format"),
                    )
                    self.append(self.btn_time)
                case "volume":
                    if "format" in module:
                        for vol in range(0, 101):
                            vol_label = self.formatter.format(
                                module["format"], volume=vol
                            )
                            self.vol_labels.append(vol_label)
                        self.btn_volume = PlayerctlButton(
                            "playerctl-volume", self.vol_labels[0]
                        )
                        self.append(self.btn_volume)
                case "title":
                    label_empty = module.get("label_empty", "")
                    scroll_speed = module.get("scroll_speed", 10)
                    scroll_width = module.get("scroll_width", 0)
                    label_format = module.get("label_format")

                    self.ticker = module.get("ticker", False)
                    self.label_title = PlayerctlButton(
                        "playerctl-title", label_empty, label_format
                    )
                    self.scroll_view = Gtk.ScrolledWindow.new()
                    self.box_title = Gtk.Box.new(Gtk.Orientation.HORIZONTAL, 0)
                    self.box_title.append(self.label_title)

                    if scroll_width > 0:
                        scroll_speed = max(1, min(scroll_speed, self.MAX_SCROLL_SPEED))
                        self.scroll_view.set_min_content_width(scroll_width)
                        self.scroll_view.set_size_request(scroll_width, -1)
                        self.scroll_view.set_policy(
                            Gtk.PolicyType.EXTERNAL, Gtk.PolicyType.NEVER
                        )
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
                        self.scroll_view.set_policy(
                            Gtk.PolicyType.NEVER, Gtk.PolicyType.NEVER
                        )
                        self.scroll_view.set_child(self.box_title)

                    self.append(self.scroll_view)

        self.set_visible(self.always_show)

    def control_play_pause(self):
        pass

    def control_next(self):
        pass

    def control_previous(self):
        pass

    def init_player(self, name):
        # choose if you want to manage the player based on the name
        # logging.debug("player appeared: %s", name.name)
        if name.name in self.player_names:

            player = Playerctl.Player.new_from_name(name)

            props = player.props

            player.connect("playback-status", self.on_status)
            player.connect("metadata", self.on_metadata)
            player.connect("volume", self.on_volume)
            player.connect("seeked", self.on_seek)
            player.connect("shuffle", self.on_shuffle)
            self.manager.manage_player(player)

            self.on_status(player, props.playback_status)
            self.on_volume(player, props.volume)
            self.on_metadata(player, props.metadata)
            self.on_shuffle(player, props.shuffle)

            if props.can_control:
                if props.can_play and props.can_pause:
                    if self.btn_play_pause is not None:
                        self.btn_play_pause.onclick_call(
                            1, self._player_call_threadsafe, "play_pause"
                        )
                else:
                    self.log.warning(
                        "player %s playback cannot be remotely started or stopped",
                        name.name,
                    )

                if props.can_go_next and props.can_go_previous:
                    if self.btn_next is not None:
                        self.btn_next.onclick_call(
                            1, self._player_call_threadsafe, "next"
                        )

                    if self.btn_previous is not None:
                        self.btn_previous.onclick_call(
                            1, self._player_call_threadsafe, "previous"
                        )

                if self.btn_shuffle is not None:
                    self.btn_shuffle.onclick_call(1, self.toggle_shuffle_threadsafe)

            else:
                self.log.warning("player %s cannot be controlled", name.name)

            self.player = player

    def toggle_shuffle_threadsafe(self):
        if self.player is not None:
            self._player_call_threadsafe("set_shuffle", not self.player.props.shuffle)

    def _player_call_threadsafe(self, method: str, *args, **kwargs):

        def _call_method(method: str, *args, **kwargs):
            if self.player is not None:
                try:
                    getattr(self.player, method)(*args, **kwargs)
                except GLib.GError:
                    self.log.error("cannot perform action: %s", method)

        if self.aio_loop is not None:
            self.aio_loop.call_soon_threadsafe(_call_method, method, *args, **kwargs)

    def increment_ticker_idle(self):

        if self.ticker and self.h_adj is not None:
            curr_value = self.h_adj.get_value()

            if (curr_value + self.h_adj.get_page_size()) >= self.h_adj.get_upper():
                self.ticker_direction = -1
            elif curr_value <= 1:
                self.ticker_direction = 1

            if self.ticker_direction != 0:
                GLib.idle_add(
                    self.h_adj.set_value, curr_value + (self.ticker_direction * 10)
                )

    def on_metadata(self, player: Playerctl.Player, metadata: Gtk.GVariant):
        self.reset_time()

        if player is not None:
            self.log.debug(
                "received metadata for player: <%s>", player.props.player_name
            )
            self.player = player

            keys = metadata.keys()

            artist = "Unknown Artist"
            if "xesam:artist" in keys and metadata["xesam:artist"]:
                artist = metadata["xesam:artist"][0]

            album = "Unknown Album"
            if "xesam:album" in keys and metadata["xesam:album"]:
                album = metadata["xesam:album"]

            title = "Unknown Title"
            if "xesam:title" in keys and metadata["xesam:title"]:
                title = metadata["xesam:title"]

            if "mpris:length" in keys:
                self.t_total = int(metadata["mpris:length"] / 10**6)

            self.format_title_idle(artist=artist, album=album, title=title)

    def on_volume(self, player: Playerctl.Player, volume: float):
        vol = int(volume)
        if self.vol_labels:
            self.set_volume_idle(self.vol_labels[vol])

    def on_name_appeared(
        self, manager: Playerctl.PlayerManager, name: Playerctl.PlayerName
    ):
        self.init_player(name)

    def on_player_vanished(
        self, manager: Playerctl.PlayerManager, player: Playerctl.Player
    ):
        self.player_ready.clear()
        self.on_status(None, Playerctl.PlaybackStatus.STOPPED)

    def reset_time(self):

        def _reset():
            self._t_last = 0

        if self.aio_loop is not None:
            self.aio_loop.call_soon_threadsafe(_reset)

    def on_shuffle(self, player: Playerctl.Player, shuffle_status: bool):
        self.player = player

        if self.btn_shuffle is not None:
            if shuffle_status:
                self.btn_shuffle.set_label_idle(self.label_shuffle_on)
            else:
                self.btn_shuffle.reset_label_idle()

    def on_seek(self, player: Playerctl.Player, *_):
        self.on_status(player, player.props.playback_status)

    def on_status(self, player: Playerctl.Player, status: Playerctl.PlaybackStatus):
        self.reset_time()

        if status == Playerctl.PlaybackStatus.PLAYING:
            self.player = player
            self.set_visible(True)
            self.player_ready.set()
            self.set_play_idle(False)
        elif status == Playerctl.PlaybackStatus.PAUSED:

            self.player_ready.clear()

            if player is not None:
                raw_sec = int(player.get_position() / 10**6)
                t_min, t_sec = divmod(raw_sec, 60)
                t_tot_min, t_tot_sec = divmod(self.t_total, 60)
                self.format_time_idle(
                    current=f"{t_min}:{t_sec:02d}", total=f"{t_tot_min}:{t_tot_sec:02d}"
                )

            self.set_play_idle(True)
        else:
            self.set_visible(self.always_show)
            self.player_ready.clear()
            self.format_title_idle()
            self.format_time_idle()
            self.set_volume_idle(None)
            self.t_total = 0

    def format_time_idle(self, **kwargs):
        if self.btn_time is not None:
            if not kwargs:
                self.btn_time.reset_label_idle()
            else:
                self.btn_time.format_label_idle(**kwargs)

    def set_play_idle(self, is_play: bool):
        if self.btn_play_pause is not None:
            if is_play:
                self.btn_play_pause.set_label_idle(self.label_play)
            else:
                self.btn_play_pause.set_label_idle(self.label_pause)

    def format_title_idle(self, **kwargs):
        if self.label_title is not None:
            if not kwargs:
                self.label_title.reset_label_idle()
            else:
                self.label_title.format_label_idle(**kwargs)

    def set_volume_idle(self, volume_str: str | None):
        if self.btn_volume is not None:
            if volume_str is None:
                self.btn_volume.reset_label_idle()
            else:
                self.btn_volume.set_label_idle(volume_str)

    async def watch_positon(self):

        while self._run:
            await self.player_ready.wait()

            self.reset_time()

            if self.player is not None:

                t_tot_min, t_tot_sec = divmod(self.t_total, 60)

                while self.player_ready.is_set():
                    if (
                        self.player.props.playback_status
                        == Playerctl.PlaybackStatus.PLAYING
                    ):

                        raw_sec = int(self.player.get_position() / 10**6)

                        if self._t_last < raw_sec:
                            self._t_last = raw_sec

                            t_min, t_sec = divmod(raw_sec, 60)

                            self.format_time_idle(
                                current=f"{t_min}:{t_sec:02d}",
                                total=f"{t_tot_min}:{t_tot_sec:02d}",
                            )

                        self.increment_ticker_idle()

                        await asyncio.sleep(self.tick)
            await asyncio.sleep(1)

    async def run(self):
        self.aio_loop = asyncio.get_running_loop()

        self.manager.connect("name-appeared", self.on_name_appeared)
        self.manager.connect("player-vanished", self.on_player_vanished)

        for name in self.manager.props.player_names:
            self.init_player(name)

        async with asyncio.TaskGroup() as tg:
            task = tg.create_task(self.watch_positon())

    def stop(self):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()
