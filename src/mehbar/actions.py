import logging
import shlex
import subprocess
import threading
from collections.abc import Callable


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
            self.args = list(args)

    def run(self):
        try:
            proc = subprocess.Popen(
                self.args,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as ex:
            logging.error("cannot execute %s: %s", self.args, ex)
        else:
            # reap the child once it exits so that it does not linger as a zombie
            threading.Thread(target=proc.wait, name="Reaper", daemon=True).start()
