import os
import sys
import logging

_LOG_LEVEL = os.getenv("MEHBAR_LOG_LEVEL", "DEBUG").upper()


class LevelAwareLoggingFormatter(logging.Formatter):

    DEFAULT_FORMAT = "[%(asctime)s] *%(levelname)s* <%(name)s>: %(message)s"

    LEVEL_FORMATS = {
        logging.DEBUG: "[%(asctime)s] *%(levelname)s* <%(name)s> (<%(threadName)s> 0x%(thread)x, <%(taskName)s>): %(message)s"
    }

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True, *, defaults=None):
        self._styles = {}
        self.datefmt = datefmt

        for levelno in logging.getLevelNamesMapping().values():
            style = logging.PercentStyle(self.LEVEL_FORMATS.get(levelno, self.DEFAULT_FORMAT),
                                         defaults=defaults)
            if validate:
                style.validate()
            self._styles[levelno] = style

    def usesTime(self):
        return True

    def formatMessage(self, record: logging.LogRecord):
        return self._styles[record.levelno].format(record)

logging.basicConfig(
    level=logging.getLevelNamesMapping().get(_LOG_LEVEL, logging.INFO)
)

for handler in logging.root.handlers:
    handler.setFormatter(LevelAwareLoggingFormatter())

# Remove '' and current working directory from the first entry
# of sys.path, if present to avoid using current directory
# in pip commands check, freeze, install, list and show,
# when invoked as python -m pip <command>
if sys.path[0] in ("", os.getcwd()):
    sys.path.pop(0)

# If we are running from a wheel, add the wheel to sys.path
# This allows the usage python pip-*.whl/pip install pip-*.whl
if __package__ == "":
    # __file__ is pip-*.whl/pip/__main__.py
    # first dirname call strips of '/__main__.py', second strips off '/pip'
    # Resulting path is the name of the wheel itself
    # Add that to sys.path so we can import pip
    path = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, path)

if __name__ == "__main__":
    from mehbar._internals import main as _main

    _main.entrypoint(sys.argv)


