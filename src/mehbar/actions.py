import logging
import shlex
import subprocess
import threading
from collections.abc import Callable

from .exceptions import BarConfigError
from .tools import child_environ


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
    """Runs a command, either a string split like a shell does or a list of
    arguments. `option` names the command in configuration errors."""

    def __init__(self, args: str | list[str], option: str = "command"):
        if isinstance(args, str):
            try:
                self.args = shlex.split(args)
            except ValueError as ex:
                raise BarConfigError(f"{option}: {ex}") from ex
        elif isinstance(args, list) and all(isinstance(arg, str) for arg in args):
            self.args = list(args)
        else:
            raise BarConfigError(f"{option} must be a string or a list of strings")

        if not self.args:
            raise BarConfigError(f"{option} is empty")

    def run(self):
        try:
            proc = subprocess.Popen(
                self.args,
                start_new_session=True,
                env=child_environ(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, ValueError) as ex:  # ValueError, e.g. for a null byte
            logging.error("cannot execute %s: %s", self.args, ex)
        else:
            # reap the child once it exits so that it does not linger as a zombie
            threading.Thread(target=proc.wait, name="Reaper", daemon=True).start()
