# mehbar
Highly customizable GPU-accelerated Wayland status bar

## Enable brightness control for DDC/CI-compatible monitors
1. add `dtparam=i2c_arm=on` to /boot/config.txt
2. load ddcci, backlight, ddcci_backlight, i2c-dev modules
3. echo "ddcci 0x37" > /sys/bus/i2c/devices/<device>/new_device, where device is the one you see in the output of ddcutil detect

# udev rules
/etc/udev/rules.d/backlight.rules:
ACTION=="add", SUBSYSTEM=="backlight", GROUP="video", MODE="0664"
udevadm control --reload-rules
udevadm trigger

## Themes
Select a theme with `theme = "<name>"` in the `[bar]` section of `config.toml`, or with `--theme`. Themes shipped with mehbar:

- `tokyo-night`: a muted dark theme, based on the [Tokyo Night](https://github.com/folke/tokyonight.nvim) palette. It uses [JetBrains Mono](https://www.jetbrains.com/lp/mono/) if installed. Ramp entries may set its `dim`, `accent`, `good`, `warning` and `critical` classes, e.g. `"[icon=battery-warning; classes=critical;]"`.

Your own `style.css` is applied on top of the theme. Rules that set colors there override the theme.

Besides the classes set in labels and ramps, widgets have the `bar-widget` class and a class named after their type, e.g. `.bar-widget.i3_mode`. The bar has `bar-top` or `bar-bottom`.

## Building themes
Theme sources live in `examples/<name>`, compile them into the package with:

    ./compile-gresources.sh /org/mehbar/themes/_base examples/_base src/mehbar/assets/_base.gresource
    ./compile-gresources.sh /org/mehbar/themes/tokyo-night examples/tokyo-night src/mehbar/assets/tokyo-night.gresource

User themes go to `~/.config/mehbar/themes/`, either as a `<name>` directory or as a `.gresource` file with the `/org/mehbar/themes/<name>` prefix.

## Icons
The base theme uses [Phosphor Icons](https://phosphoricons.com), MIT license, see `examples/_base/icons/LICENSE`.
