import argparse
import hashlib
import logging
import os
import string
from functools import partial
from importlib import resources
from pathlib import Path

#!/usr/bin/env python3
from typing import Any, Mapping, Sequence

import gi

NULL = object()


class SelectorDict(dict):
    def select(self, path: str, default: Any | None = NULL):
        result = self

        is_error = False

        parts = path.split(".")
        while parts:
            if (part := parts.pop(0)) in result:
                result = result[part]
                if not isinstance(result, dict):
                    if parts:
                        is_error = True
                    break
            else:
                is_error = True
                break
        if is_error:
            if default is not NULL:
                result = default
            else:
                raise KeyError(f"no element at path '{path}'")

        return result


class Colorize:
    RESET = "\x1b[0m"
    BLACK = "\x1b[30m"
    BLUE = "\x1b[34m"
    CYAN = "\x1b[36m"
    GREEN = "\x1b[32m"
    GREY = "\x1b[90m"
    MAGENTA = "\x1b[35m"
    RED = "\x1b[31m"
    WHITE = "\x1b[37m"  # more like LIGHT GRAY
    YELLOW = "\x1b[33m"

    BOLD = "\x1b[1m"
    BOLD_BLACK = "\x1b[1;30m"  # DARK GRAY
    BOLD_BLUE = "\x1b[1;34m"
    BOLD_CYAN = "\x1b[1;36m"
    BOLD_GREEN = "\x1b[1;32m"
    BOLD_MAGENTA = "\x1b[1;35m"
    BOLD_RED = "\x1b[1;31m"
    BOLD_WHITE = "\x1b[1;37m"  # actual WHITE
    BOLD_YELLOW = "\x1b[1;33m"

    # intense = like bold but without being bold
    INTENSE_BLACK = "\x1b[90m"
    INTENSE_BLUE = "\x1b[94m"
    INTENSE_CYAN = "\x1b[96m"
    INTENSE_GREEN = "\x1b[92m"
    INTENSE_MAGENTA = "\x1b[95m"
    INTENSE_RED = "\x1b[91m"
    INTENSE_WHITE = "\x1b[97m"
    INTENSE_YELLOW = "\x1b[93m"

    BACKGROUND_BLACK = "\x1b[40m"
    BACKGROUND_BLUE = "\x1b[44m"
    BACKGROUND_CYAN = "\x1b[46m"
    BACKGROUND_GREEN = "\x1b[42m"
    BACKGROUND_MAGENTA = "\x1b[45m"
    BACKGROUND_RED = "\x1b[41m"
    BACKGROUND_WHITE = "\x1b[47m"
    BACKGROUND_YELLOW = "\x1b[43m"

    INTENSE_BACKGROUND_BLACK = "\x1b[100m"
    INTENSE_BACKGROUND_BLUE = "\x1b[104m"
    INTENSE_BACKGROUND_CYAN = "\x1b[106m"
    INTENSE_BACKGROUND_GREEN = "\x1b[102m"
    INTENSE_BACKGROUND_MAGENTA = "\x1b[105m"
    INTENSE_BACKGROUND_RED = "\x1b[101m"
    INTENSE_BACKGROUND_WHITE = "\x1b[107m"
    INTENSE_BACKGROUND_YELLOW = "\x1b[103m"


def next_prime(num: int, offset: int = 0):
    num += offset

    while not is_prime(num):
        num += 1
    return num


def is_prime(num: int):
    if num <= 1:
        ret = False
    else:
        ret = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                ret = False
                break
    return ret


def overlay_dict_r(
    bottom: dict[Any, Any], top: dict[Any, Any], max_depth: int = 10, depth: int = 0
):
    if depth > max_depth:
        raise ValueError(f"maximum nesting depth exceeded: {max_depth}")

    for ktop, vtop in top.items():
        if isinstance(vtop, dict):
            if ktop not in bottom or not isinstance(bottom[ktop], dict):
                bottom[ktop] = {}
            overlay_dict_r(bottom[ktop], vtop, max_depth, depth + 1)
        else:
            bottom[ktop] = vtop


def md5sum_sync(fpath: str | Path) -> str:
    hash = hashlib.md5()
    with open(fpath, "rb") as fhandle:
        reader = partial(fhandle.read, 128 * hash.block_size)
        for chunk in iter(reader, b""):
            hash.update(chunk)
    return hash.hexdigest()


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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(seconds={self.tot_secs})"


def get_config_home() -> Path:
    if (cfg_home := os.getenv("XDG_CONFIG_HOME")) is None:
        cfg_home = Path.home() / ".config"
    else:
        cfg_home = Path(cfg_home)
    return cfg_home / "mehbar"


def get_asset(asset: str | Path) -> Path | None:
    asset_file = None
    with resources.as_file(resources.files()) as mod_dir:
        asset_file_ = mod_dir / "assets" / asset

        if asset_file_.is_file():
            asset_file = asset_file_
    return asset_file


def get_system_cs() -> str:
    ret = "system"

    try:
        from gi.repository import Gio

        gsettings_schema = "org.gnome.desktop.interface"

        if Gio.SettingsSchemaSource.get_default().lookup(gsettings_schema) is not None:
            gsettings = Gio.Settings.new(gsettings_schema)

            gsettings_cs = gsettings.get_string("color-scheme")
            if gsettings_cs == "prefer-dark":
                ret = "dark"
            elif gsettings_cs == "prefer-light":
                ret = "light"
    except ImportError:
        pass

    if ret is None:
        try:
            gi.require_version("Adw", "1")
            from gi.repository import Adw

            style_mgr = Adw.StyleManager.get_default()
            ret = "dark" if style_mgr.get_dark() else "light"
        except (ImportError, ValueError):
            pass

    return ret


class OptionalFormatter(string.Formatter):
    """Like the default stripng formatter that you know and love but silently
    skips missing fields.
    """

    def get_fields(self, format_string) -> list[str]:
        ret = []
        for _, fld, _, _ in self.parse(format_string):
            if fld is not None:
                ret.append(fld)

        return ret

    def vformat(
        self,
        format_string: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> str:

        if args:
            raise ValueError("non-keyword arguments are not supported")

        unparsed = str()

        for literal, fld, spec, _ in self.parse(format_string):
            if fld is None or kwargs.get(fld) is None:
                unparsed += literal
            elif fld in kwargs and str(kwargs[fld]):
                unparsed += literal + "{" + fld
                if spec is not None:
                    unparsed += ":" + spec
                unparsed += "}"

        return super().vformat(unparsed, args, kwargs)


class LevelAwareLoggingFormatter(logging.Formatter):
    DEFAULT_FORMAT = "[%(asctime)s] *%(levelname)s*: %(message)s"

    LEVEL_FORMATS = {
        logging.DEBUG: "[%(asctime)s] *%(levelname)s* <%(name)s> (<%(threadName)s> 0x%(thread)x): %(message)s"
    }

    NO_EXC_INFO = (None, None, None)

    def __init__(
        self, fmt=None, datefmt=None, style="%", validate=True, *, defaults=None
    ):
        self._styles = {}
        self.datefmt = datefmt

        for levelno in logging.getLevelNamesMapping().values():
            level_fmt = self.LEVEL_FORMATS.get(levelno, self.DEFAULT_FORMAT)
            level_style = logging.PercentStyle(level_fmt, defaults=defaults)

            if validate:
                level_style.validate()
            self._styles[levelno] = level_style

    def usesTime(self):
        return True

    def formatException(self, ei):
        return super().formatException(ei) if ei != self.NO_EXC_INFO else None

    def formatMessage(self, record: logging.LogRecord):
        return self._styles[record.levelno].format(record)
