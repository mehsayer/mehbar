# mehbar
Highly customizable GPU-accelerated Wayland status bar



## Enable brightness control for DDC/CI-compatible monitors
1. add `dtparam=i2c_arm=on` to /boot/config.txt
2. load ddcci, backlight, ddcci_backlight, i2c-dev modules
3. echo "ddcci 0x37" > /sys/bus/i2c/devices/<device>/new_device, whre device is the one you see in the output of ddcutil detect


# udev rules
/etc/udev/rules.d/backlight.rules:
ACTION=="add", SUBSYSTEM=="backlight", GROUP="video", MODE="0664"
udevadm control --reload-rules
udevadm trigger
