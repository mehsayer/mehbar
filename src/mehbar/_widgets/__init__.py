"""Built-in widgets.

Widget modules are imported on demand, so that missing optional dependencies
(i3ipc, psutil, pulsectl-asyncio, Playerctl) only disable widgets that need
them.
"""

import importlib

from mehbar.exceptions import BarConfigError
from mehbar.widget import BarWidget

# widget type -> (module, class name)
WIDGET_TYPES: dict[str, tuple[str, str]] = {
    "application": ("application", "WidgetApplication"),
    "backlight": ("backlight", "WidgetBacklight"),
    "battery": ("battery", "WidgetBattery"),
    "bluetooth-status": ("bluetooth", "WidgetBluetoothStatus"),
    "cpu_fq": ("cpu_fq", "WidgetCPUFrequency"),
    "cpu_usage": ("cpu_usage", "WidgetCPUUsage"),
    "datetime": ("datetime", "WidgetDateTime"),
    "disk_usage": ("disk", "WidgetDiskUsage"),
    "exec_repeat": ("exec", "WidgetExecRepeat"),
    "exec_tail": ("exec", "WidgetExecTail"),
    "fan_speed": ("fan_speed", "WidgetFanSpeed"),
    "file": ("file", "WidgetFile"),
    "i3_kblayout": ("i3_keyboard", "WidgetI3KeyboardLayout"),
    "i3_mode": ("i3_mode", "WidgetI3Mode"),
    "i3_scratchpad": ("i3_scratchpad", "WidgetI3Scratchpad"),
    "i3_window": ("i3_window", "WidgetI3Window"),
    "i3_workspaces": ("i3_workspaces", "WidgetI3Workspaces"),
    "memory_usage": ("memory", "WidgetMemoryUsage"),
    "network_rate": ("network_rate", "WidgetNetworkRate"),
    "playerctl": ("playerctl", "WidgetPlayerCtl"),
    "pulse_volume": ("pulse_volume", "WidgetPulseVolume"),
    "session": ("session", "WidgetSession"),
    "static": ("static", "WidgetStatic"),
    "temperature": ("temperature", "WidgetTemperature"),
    "wifi": ("wifi", "WidgetWifi"),
    "wired": ("wired", "WidgetWired"),
}


def get_widget_class(wtype: str) -> type[BarWidget]:
    if (entry := WIDGET_TYPES.get(wtype)) is None:
        types = ", ".join(sorted(WIDGET_TYPES))
        raise BarConfigError(f"widget type '{wtype}' is not one of: {types}")

    module_name, class_name = entry

    try:
        module = importlib.import_module(f"{__name__}.{module_name}")
    except (ImportError, ValueError) as ex:
        # ValueError is raised by gi.require_version()
        raise BarConfigError(f"widget type '{wtype}' is not available: {ex}") from ex

    widget_cls = getattr(module, class_name)

    if getattr(widget_cls, "TYPE", None) != wtype:
        raise RuntimeError(f"widget class '{class_name}' is not of type '{wtype}'")

    return widget_cls


__all__ = ["WIDGET_TYPES", "get_widget_class"]
