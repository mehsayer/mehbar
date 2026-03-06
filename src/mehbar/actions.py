from enum import Enum
from collections.abc import Callable

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
