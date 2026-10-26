# mehbar

A highly customizable status bar for Wayland compositors, built with GTK 4 and
gtk4-layer-shell. Widgets show system state (CPU, memory, sensors, battery,
network, Bluetooth, volume, media players, i3/sway workspaces and more), and
can be styled with CSS themes.

## Requirements

- Python 3.14 or later
- GTK 4 and [gtk4-layer-shell](https://github.com/wmww/gtk4-layer-shell), with
  their GObject introspection data
- A compositor supporting the layer shell protocol, e.g. sway, Hyprland or river
- Optional, per widget:
  - `i3_*` widgets: sway or i3 IPC, and the `i3ipc` package
  - `pulse_volume`: PulseAudio or PipeWire, and the `pulsectl-asyncio` package
  - `playerctl`: [Playerctl](https://github.com/altdesktop/playerctl) with its
    introspection data
  - `bluetooth-status`, `wifi`, `wired`: the respective D-Bus services, e.g.
    BlueZ, iwd or NetworkManager

A widget whose optional dependency is missing is disabled; the rest of the bar
works.

## Installation

    python -m venv venv
    venv/bin/pip install '.[all]'

The `all` extra installs every optional Python dependency, see
`pyproject.toml` for the others: `i3`, `pulse`, `yaml` and `json5`.

## Running

    mehbar [-c CFG_DIR] [-t THEME] [-s COLOR_SCHEME] [-L LEVEL] [-E]

| Option | Description |
|---|---|
| `-c`, `--config-dir` | Configuration directory, `$XDG_CONFIG_HOME/mehbar` by default. Without one, the built-in configuration is used. |
| `-t`, `--theme` | Theme name, see [Themes](#themes) |
| `-s`, `--color-scheme` | `light`, `dark` or `system`, the desktop preference |
| `-L`, `--log-level` | `debug`, `info`, `warning`, `error` or `critical` |
| `-E`, `--exception-info` | Log tracebacks of errors |

Command line options override configuration files. Only one instance runs at a
time: starting another one logs a warning and exits, stop the running bar to
apply new options. For sway, add to its configuration:

    exec mehbar

## Configuration

The configuration directory contains `config.toml` and, optionally,
`style.css`. Configuration may also be written as `config.json`, and, with the
respective extras installed, `config.jsonc`, `config.json5` or `config.yaml`.

Settings are layered: the built-in base configuration, then the theme, then
the user configuration, later layers override earlier ones. Widgets of the
base configuration are only used if neither the theme nor the user
configuration define any. For the active color scheme, `config-dark.toml` or
`config-light.toml`, and `style-dark.css` or `style-light.css`, are applied
on top of each layer.

Without a configuration, the [base configuration](examples/_base/config.toml)
is used: workspaces, the focused window, CPU, memory, temperature, disk,
network, volume, battery and the clock. It needs no adjustment: network
interfaces and backends, and the CPU temperature sensor, are detected, and
widgets of missing hardware or services are hidden. The base theme is dark,
and so is the color scheme when the desktop has no preference.

### The bar

```toml
[bar]
position = "bottom"     # top or bottom
layer = "top"           # top or bottom
height = 32
width = 0               # more than 256 for a fixed, centered width
gaps = [8, 8, 8, 8]     # top, left, bottom, right; or vertical, horizontal
homogeneous = true      # equally wide start, center and end sections
icon_size = 20
theme = "tokyo-night"
color_scheme = "dark"   # light, dark or system
tooltip_trigger = "hover"  # hover or click, see Tooltips
```

### Widgets

Widgets are placed in the `start`, `center` and `end` sections. Each one has
a unique name and a type:

```toml
[center.cpu]
type = "cpu_usage"
interval = 3
label = "[icon=cpu;]{percent}%"
```

Options common to all widgets:

| Option | Description |
|---|---|
| `type` | Widget type, see [Widget types](#widget-types) |
| `label` | Text, with format fields and an optional [specification](#labels) |
| `ramp` | Labels shown for levels of the widget value, e.g. icons from empty to full |
| `max_ramp_level` | The value at which the last ramp entry is shown |
| `interval` | Seconds between updates of polling widgets |
| `relax_interval` | `true` by default, see below |
| `onclick` | Commands run on click, the first for the left button, the second for the middle one, the third for the right one; `""` skips a button |
| `onscroll` | Two commands, run on scrolling up and down; `""` does nothing |
| `width_chars`, `max_width_chars` | Label width, longer labels are ellipsized |
| `tooltip`, `tooltip_trigger` | See [Tooltips](#tooltips) |

Commands are either a string, split like a shell would, or a list of
arguments, e.g. `onclick = ["pavucontrol", "", ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"]]`.

To spread the load, widgets polling at the same interval do not wake up
together: intervals are stretched by at most the interval itself and 2.5
seconds, e.g. ten widgets set to 3 seconds update every 3 to 5.5 seconds, and
widgets sharing an interval wake up at different times. Set `relax_interval =
false` to keep the exact interval. `datetime` always keeps it.

A widget that fails is disabled, and the error is logged, the rest of the bar
keeps working. Temporary conditions do not disable widgets: a widget is
hidden while, e.g., its device is unplugged or its service restarts, and
shown again once it is back. Widgets reconnect to sway and to the sound
server, and pick up restarted D-Bus services. Widgets of hardware or
services that are missing altogether are hidden: `battery` without
batteries, `wifi` and `wired` without interfaces, `i3_*` widgets without i3
or sway.

### Tooltips

The `tooltip` option is formatted like the label, with the same fields and
specification, and may span several lines:

```toml
tooltip = "[icon=memory;]{used_gib} of {total_gib} GiB used"
```

Tooltips appear on hover, or, with `tooltip_trigger = "click"` in `[bar]`
or in a widget, on a left click; widgets whose left click does something
else, e.g. `application` or an `onclick` command, show them on hover. They
are made when they appear and do not follow later changes. Many widgets have
a default tooltip, `tooltip = ""` turns it off. The `datetime` tooltip is a
calendar, unless `tooltip` is set. A `tooltip` field in a label or ramp
specification overrides the option.

Unknown `[bar]` options, e.g. misspelled ones, are reported in the log.

### Labels

Labels are Python format strings: `{percent}%`, `{datetime:%H:%M}`,
`{used_gib:.1f}`. Missing fields are shown empty. A label may start with a
bracketed specification of `;`-terminated fields:

```
[icon=cpu; classes=busy; tooltip=CPU {percent}%;]{percent}%
```

| Field | Description |
|---|---|
| `icon=<name>` | A preloaded icon, see [Icons](#icons). `<name>>` places it after the label. `!none` hides the icon, `!default` keeps the label icon (useful in ramps). |
| `classes=<class> ...` | CSS classes of the widget, `classes:widget=..., icon=..., label=...` sets them per part. `!none` removes them. |
| `tooltip=<text>` | Tooltip, may contain format fields |

Ramp entries use the same syntax, the selected entry overrides the label's
icon, classes and tooltip, and its text is available as the `{ramp}` field. A
`{ramp}` field in the middle of the label places the ramp icon there:

```toml
ramp = ["[icon=speaker-x;]", "[icon=speaker-low;]", "[icon=speaker-high;]"]
label = "{percent}%"
```

### Icons

Icons are preloaded by name, from the theme, the GTK icon theme or files:

```toml
[preload-icons]
cpu = "theme://cpu-symbolic.svg"         # icons/ of the theme, or the base theme
firefox = "icontheme://firefox"          # the GTK icon theme
logo = "file:///home/me/pictures/logo.svg"
```

Symbolic icons, named `*-symbolic.svg`, are recolored by the CSS `color`
property. The base theme uses [Phosphor Icons](https://phosphoricons.com), MIT
license, see `examples/_base/icons/LICENSE`, and preloads all of its icons.

### Widget types

Fields are available in labels. Polling widgets need an `interval`.

| Type | Options | Fields | Ramp |
|---|---|---|---|
| `application` | `app_id` (desktop entry, e.g. `firefox.desktop`) or `app_name`, `files` | `name`, `id` | |
| `backlight` | `driver` (`acpi` or `ddcci`), `device`, `step` | `percent` | brightness |
| `battery` | | `percent`, `timeleft`, `status`, `plugged` | pairs of on battery, plugged in |
| `bluetooth-status` | | `status`, `name`, `alias`, `battery_percent`, `volume_percent` | off, on, connected |
| `cpu_fq` | | `fq`, `fq_min`, `fq_max` (GHz), `percent` | frequency |
| `cpu_usage` | | `percent` | usage |
| `datetime` | | `datetime` | hours of the day |
| `disk_usage` | `path` | `percent`, `used_gib`, `avail_gib`, `total_gib` | usage |
| `exec_repeat` | `cmdline`, `timeout` | see below | `ramp_level` |
| `exec_tail` | `cmdline`, `max_lps` | see below | `ramp_level` |
| `fan_speed` | `path`, `max_speed` | `rpm`, `<chip>/<label>` without `path` | speed |
| `file` | `path` | see below | `ramp_level` |
| `i3_kblayout` | `rewrite` | `layout` | |
| `idle_inhibitor` | `active` | `status` | off, on |
| `i3_mode` | `always_show`, `rewrite` | `mode` | |
| `i3_scratchpad` | `always_show` | `count` | count |
| `i3_window` | `always_show`, `rewrite` | `title` | |
| `i3_workspaces` | `always_show`, `max_workspaces`, `scroll_width`, `scroll_speed`, `rewrite` | `name`, `num` | empty, normal, focused, urgent |
| `memory_usage` | | `percent`, `used_mib`, `used_gib`, `avail_mib`, `avail_gib`, `total_mib`, `total_gib` | usage |
| `network_rate` | `iface` (all by default), `conv_map` | `rate_rx`, `rate_tx`, `unit_rx`, `unit_tx` | |
| `playerctl` | `player_names`, `modules`, `always_show`, `tick_ms`, `seek_offset` | | |
| `pulse_volume` | `sink_name`, `max_volume`, `volume_delta` | `percent`, `muted` | muted, then volume |
| `session` | | `username`, `uid`, `hostname`, `fqdn`, `uptime` | |
| `static` | | | |
| `temperature` | `path`, `max_temp` | `temp`, `<chip>/<label>` without `path` | temperature |
| `weather` | `location` or `latitude` and `longitude`, `provider`, `api_key`, `units` | `temp`, `feels_like`, `humidity`, `wind_speed`, `wind_direction`, `condition`, `location`, `temp_unit`, `speed_unit` | conditions, see below |
| `wifi` | `backend`, `iface` | `ssid`, `percent`, `rssi`, `security`, `ipv4`, `ipv6`, `hwaddr`, `iface` | no signal, then signal |
| `wired` | `backend`, `iface` | `ipv4`, `ipv6`, `hwaddr`, `name`, `iface` | down, no link, connected |

Notes:

- `exec_repeat` runs `cmdline` every interval and shows the first line of
  its output, `exec_tail` shows the latest line of a long running command,
  `file` the last line of a file. A string `cmdline` is run by the shell. A
  line containing a JSON object provides fields, and `ramp_level` selects the
  ramp entry; any other line is the `label` field.
- `temperature` and `fan_speed` show the `path` file as `{temp}` and
  `{rpm}`. Without `path`, `{temp}` is the CPU temperature, `{rpm}` the speed
  of the fastest fan, and other fields name sensors as `psutil` reports them,
  e.g. `{k10temp/Tctl}`. Sensors repeated by chips of the same name, e.g. two
  NVMe drives, are numbered: `{nvme/Composite}`, `{nvme/Composite#2}`.
- `wifi` and `wired` use the `iface` interface, or the first one found.
  `wifi` backends: `NetworkManager`, `iwd`, `connman`, `wpa_supplicant` or
  `unmanaged`; `wired` backends: `NetworkManager`, `connman` or `unmanaged`.
  By default, `auto` picks the network service that is running. Signal
  strength is mapped to percent as NetworkManager does, -100 to -40 dBm.
- `network_rate` units are bytes per second: `B/s`, `KiB/s`, `MiB/s`.
  `conv_map` maps other unit names to their size in bytes.
- `rewrite` maps regular expressions to replacements, the first expression
  matching at the start of the text wins, e.g. `"Eng.+" = "EN"`.
- `playerctl` modules: `previous`, `next`, `seek_back`, `seek_forward`,
  `shuffle`, `play_pause`, `title` (`{artist}`, `{album}`, `{title}`),
  `time` (`{current}`, `{total}`) and `volume` (`{percent}`), see
  `DEFAULT_MODULES` in `src/mehbar/_widgets/playerctl.py`. Without
  `player_names`, it follows the player that plays.
- `pulse_volume`: scrolling changes the volume, a right click toggles mute.
- `idle_inhibitor` keeps the screen from blanking and locking while it is
  on, a click turns it on or off, `active = true` turns it on at start. It
  uses the Wayland idle inhibit protocol, which sway supports.
- `weather` providers:
  - `open-meteo`, the default, [Open-Meteo](https://open-meteo.com), free
    for non-commercial use, no key;
  - `met.no`, the [Norwegian Meteorological Institute](https://api.met.no),
    free, no key;
  - `openweathermap`, [OpenWeatherMap](https://openweathermap.org), needs
    an `api_key`, which is free for up to 1000 calls a day, sign up at
    openweathermap.org.

  `location` is a place name, looked up with the Open-Meteo geocoding API.
  `units` are `metric`, the default, or `imperial`. The weather is
  requested at most every 15 minutes, failed requests included, also across
  restarts of the bar: it is cached in `$XDG_CACHE_HOME/mehbar`, and shorter
  intervals show the cached weather. Ramp entries are for the conditions
  clear, partly cloudy, cloudy, fog, drizzle, rain, snow and thunderstorm;
  eight more entries, if given, are used at night:

  ```toml
  [end.weather]
  type = "weather"
  location = "Berlin"
  interval = 900
  ramp = ["[icon=sun;]", "[icon=cloud-sun;]", "[icon=cloud;]", "[icon=cloud-fog;]",
          "[icon=drop;]", "[icon=cloud-rain;]", "[icon=cloud-snow;]", "[icon=cloud-lightning;]"]
  label = "{temp}{temp_unit}"
  ```

## Themes

Select a theme with `theme = "<name>"` in the `[bar]` section, or with
`--theme`. Themes shipped with mehbar:

- `tokyo-night`: a muted dark theme, based on the
  [Tokyo Night](https://github.com/folke/tokyonight.nvim) palette. It uses
  [JetBrains Mono](https://www.jetbrains.com/lp/mono/) if installed. Ramp
  entries may set its `dim`, `accent`, `good`, `warning` and `critical`
  classes, e.g. `"[icon=battery-warning; classes=critical;]"`.

Your own `style.css` is applied on top of the theme, rules that set colors
there override the theme. User themes go to `~/.config/mehbar/themes/`,
either as a `<name>` directory, laid out like `examples/tokyo-night`, or as a
`<name>.gresource` bundle with the `/org/mehbar/themes/<name>` prefix.

CSS classes and names:

| Selector | Element |
|---|---|
| `.bar-horizontal`, `.bar-top`, `.bar-bottom` | the bar |
| `#start`, `#center`, `#end` | sections |
| `.bar-widget`, `.<type>` (e.g. `.bar-widget.i3_mode`), `#<name>` | widgets |
| `.bar-widget-label`, `.bar-widget-icon` | parts of widgets |
| `.bar-widget-icon-start`, `-end`, `-middle` | icons by position |
| `.i3-workspace` with `.focused`, `.visible`, `.urgent`, `.previous`, `.empty` | workspace buttons |
| `.playerctl`, `.playerctl-button`, `#playerctl-<module>` | player controls |
| `.bar-tooltip`, `.bar-tooltip-label`, `.bar-tooltip-icon`, `.bar-tooltip-popover` | tooltips |

### Building themes

Theme sources live in `examples/<name>`. After changing them, rebuild the
bundles in the package:

    ./compile-gresources.sh /org/mehbar/themes/_base examples/_base src/mehbar/assets/_base.gresource
    ./compile-gresources.sh /org/mehbar/themes/tokyo-night examples/tokyo-night src/mehbar/assets/tokyo-night.gresource

The script needs `glib-compile-resources` and, optionally, `svgo` to
optimize icons. It never modifies the sources. A test checks that the
bundles are up to date.

## Backlight

The `acpi` driver controls `/sys/class/backlight` devices. Reading needs no
permissions; brightness is set through systemd-logind (elogind) when the
device is not writable. Alternatively, a udev rule makes it writable for the
`video` group, `/etc/udev/rules.d/backlight.rules`:

    ACTION=="add", SUBSYSTEM=="backlight", GROUP="video", MODE="0664"

The `ddcci` driver talks to external monitors over DDC/CI, it needs the
`i2c-dev` kernel module and access to `/dev/i2c-*`. It probes the DDC buses
of GPU connectors first, skips SMBus and other buses that never connect
displays, and never takes over addresses used by kernel drivers, e.g. of
memory module EEPROMs. Monitors handled by the
`ddcci_backlight` kernel module appear as `/sys/class/backlight` devices,
use the `acpi` driver for them.

## Development

    venv/bin/pip install -e '.[dev]'
    venv/bin/python -m pytest
    ruff check src tests && ruff format --check src tests
    codespell src tests examples README.md

Tests marked `display` need a Wayland session, they are skipped without one.

## License

MIT, see `LICENSE`. Icons: Phosphor Icons, MIT, see
`examples/_base/icons/LICENSE`.
