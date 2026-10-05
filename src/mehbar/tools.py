import argparse
import logging
import os
import string
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any


def next_prime(num: int, offset: int = 0) -> int:
    num += offset

    while not is_prime(num):
        num += 1
    return num


def is_prime(num: int) -> bool:
    if num <= 1:
        return False

    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True


def overlay_dict_r(
    bottom: dict[Any, Any], top: Mapping[Any, Any], max_depth: int = 10, depth: int = 0
):
    """Recursively merges `top` into `bottom` in place, values from `top` win."""
    if depth > max_depth:
        raise ValueError(f"maximum nesting depth exceeded: {max_depth}")

    for ktop, vtop in top.items():
        if isinstance(vtop, Mapping):
            if not isinstance(bottom.get(ktop), dict):
                bottom[ktop] = {}
            overlay_dict_r(bottom[ktop], vtop, max_depth, depth + 1)
        else:
            bottom[ktop] = vtop


class ArgumentsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Help message formatter which adds default values that aren't
    `NoneType` or `bool` to argument help."""

    def add_argument(self, action: argparse.Action):
        if action.default is None or isinstance(action.default, bool):
            action.default = argparse.SUPPRESS
        super().add_argument(action)

    def _fill_text(self, text, width, indent):
        return "".join(indent + line for line in text.splitlines(keepends=True))


class FormattableTimeDelta:
    def __init__(self, seconds: float | int):
        self.tot_secs = int(seconds)
        self.tot_mins, self.secs = divmod(self.tot_secs, 60)
        self.tot_hrs, self.mins = divmod(self.tot_mins, 60)
        self.days, self.hrs = divmod(self.tot_hrs, 24)

    def strftime(self, format_str: str) -> str:
        fmt = []
        push = fmt.append

        i, n = 0, len(format_str)

        while i < n:
            ch = format_str[i]
            i += 1
            if ch == "%":
                if i < n:
                    ch = format_str[i]
                    i += 1
                    match ch:
                        case "s":
                            push("%d" % self.secs)
                        case "S":
                            push("%02d" % self.secs)
                        case "a":
                            push("%d" % self.tot_secs)
                        case "A":
                            push("%02d" % self.tot_secs)
                        case "m":
                            push("%d" % self.mins)
                        case "M":
                            push("%02d" % self.mins)
                        case "b":
                            push("%d" % self.tot_mins)
                        case "B":
                            push("%02d" % self.tot_mins)
                        case "h":
                            push("%d" % self.hrs)
                        case "H":
                            push("%02d" % self.hrs)
                        case "i":
                            push("%d" % self.tot_hrs)
                        case "I":
                            push("%02d" % self.tot_hrs)
                        case "d":
                            push("%d" % self.days)
                        case "D":
                            push("%02d" % self.days)
                        case _:
                            push("%")
                            push(ch)
                else:
                    push("%")
            else:
                push(ch)

        return "".join(fmt)

    def __format__(self, format_str: str) -> str:
        return self.strftime(format_str) if format_str else self.__str__()

    def __str__(self) -> str:
        return self.strftime("%I:%M:%S")

    def __int__(self) -> int:
        return self.tot_secs

    def __eq__(self, other) -> bool:
        if isinstance(other, FormattableTimeDelta):
            return self.tot_secs == other.tot_secs
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.tot_secs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(seconds={self.tot_secs})"


def get_config_home() -> Path:
    if cfg_home := os.getenv("XDG_CONFIG_HOME"):
        return Path(cfg_home) / "mehbar"
    return Path.home() / ".config" / "mehbar"


@lru_cache(maxsize=256)
def _parse_format(format_string: str) -> tuple[tuple[str, str | None, str, str], ...]:
    return tuple(string.Formatter().parse(format_string))


class OptionalFormatter(string.Formatter):
    """Like the default string formatter that you know and love, but silently
    renders missing and `None` fields as empty strings. Values that do not
    accept the format specification are rendered with `str()`.
    """

    _MISSING = object()

    def get_fields(self, format_string: str | None) -> list[str]:
        if not format_string:
            return []
        return [fld for _, fld, _, _ in _parse_format(format_string) if fld]

    def _lookup(self, field: str, kwargs: Mapping[str, Any]) -> Any:
        # Plain keys first: sensor names like 'k10temp/Tctl' or ones containing
        # dots are not valid attribute paths
        if (value := kwargs.get(field, self._MISSING)) is not self._MISSING:
            return value
        try:
            return self.get_field(field, (), kwargs)[0]
        except (KeyError, AttributeError, IndexError, TypeError, ValueError):
            return None

    def vformat(
        self,
        format_string: str | None,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> str:  # type: ignore[override]
        if args:
            raise ValueError("non-keyword arguments are not supported")

        if not format_string:
            return ""

        out = []

        for literal, field, spec, conv in _parse_format(format_string):
            out.append(literal)

            if field is None:
                continue

            if (value := self._lookup(field, kwargs)) is None:
                continue

            if conv:
                value = self.convert_field(value, conv)

            if spec and "{" in spec:
                spec = self.vformat(spec, args, kwargs)

            try:
                out.append(self.format_field(value, spec))
            except (ValueError, TypeError):
                out.append(str(value))

        return "".join(out)


class LevelAwareLoggingFormatter(logging.Formatter):
    DEFAULT_FORMAT = "[%(asctime)s] *%(levelname)s*: %(message)s"

    LEVEL_FORMATS = {
        logging.DEBUG: "[%(asctime)s] *%(levelname)s* <%(name)s> (<%(threadName)s> 0x%(thread)x): %(message)s"
    }

    NO_EXC_INFO = (None, None, None)

    def __init__(
        self, fmt=None, datefmt=None, style="%", validate=True, *, defaults=None
    ):
        super().__init__(
            self.DEFAULT_FORMAT, datefmt, style, validate, defaults=defaults
        )
        self._styles = {}

        for levelno in logging.getLevelNamesMapping().values():
            level_fmt = self.LEVEL_FORMATS.get(levelno, self.DEFAULT_FORMAT)
            level_style = logging.PercentStyle(level_fmt, defaults=defaults)

            if validate:
                level_style.validate()
            self._styles[levelno] = level_style

    def usesTime(self):
        return True

    def formatException(self, ei):
        return super().formatException(ei) if ei != self.NO_EXC_INFO else ""

    def formatMessage(self, record: logging.LogRecord):
        return self._styles.get(record.levelno, self._style).format(record)


class ExceptionInfoFilter(logging.Filter):
    """Attaches the exception currently being handled, if any, to every
    record that does not carry exception information already."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.exc_info and (exc := sys.exception()) is not None:
            record.exc_info = (type(exc), exc, exc.__traceback__)
        return True
