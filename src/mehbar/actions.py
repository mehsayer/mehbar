import shlex
import subprocess
from collections.abc import Callable

from gi.repository import Gtk


class GestureMouseClick(Gtk.GestureClick, Gtk.GestureSingle):
    pass


class ActionInterface:
    def run(self):
        raise NotImplementedError()


class CallableAction(ActionInterface):
    def __init__(self, func: Callable, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.func(*self.args, **self.kwargs)


class ExecAction(ActionInterface):
    def __init__(self, args: str | list[str]):

        if isinstance(args, str):
            self.args = shlex.split(args)
        else:
            self.args = args

    def run(self):
        subprocess.Popen(
            self.args,
            start_new_session=True,
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
