from collections.abc import Callable
from enum import Enum

from gi.repository import Gtk


class GestureMouseClick(Gtk.GestureClick, Gtk.GestureSingle):
    pass


class Action:
    def run(self):
        pass


class CallableAction(Action):
    def __init__(self, func: Callable, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.func(*self.args, **self.kwargs)


class SinkAction(Enum):
    VOLUME = 1
    MUTE = 2
