import argparse
import logging
import os
import sys
from functools import partial
from pathlib import Path

from mehbar.tools import LevelAwareLoggingFormatter, get_config_home, get_system_cs

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
    log_level_mapping = logging.getLevelNamesMapping()
    log_levels = [name.lower() for name in log_level_mapping.keys()]
    color_schemes = ["light", "dark", "system"]

    config_home = get_config_home()

    parser = argparse.ArgumentParser(
        prog="mehbar",
        description="Mehbar, a highly customizable status bar for Linux",
        epilog="Copyright (c) 2026, Mehsayer",
        suggest_on_error=True,
    )

    parser.add_argument(
        "-c",
        "--config-dir",
        dest="config_dir",
        type=Path,
        default=config_home,
        help=f"configuration directory path, defaults to '{config_home}'",
    )

    parser.add_argument(
        "-L",
        "--log-level",
        choices=log_levels,
        dest="log_level",
        metavar="LEVEL",
        default="debug",
        help=f"log level, one of {', '.join([f"'{s}'" for s in log_levels])}",
    )

    parser.add_argument(
        "-E",
        "--exception-info",
        action="store_true",
        dest="exc_info",
        help="print exception information to stdout",
    )
    parser.add_argument(
        "-t",
        "--theme",
        default=None,
        dest="theme",
        type=str,
        help="theme name",
    )
    parser.add_argument(
        "-s",
        "--color-scheme",
        choices=color_schemes,
        default=get_system_cs(),
        metavar="COLOR_SCHEME",
        dest="color_scheme",
        help=f"color scheme, one of {', '.join([f"'{s}'" for s in color_schemes])}",
    )
    args = parser.parse_args()

    if not args.config_dir.is_dir():
        parser.error(f"'{args.config_dir}' is not an existing directory")

    kwargs = vars(args)
    log_level = kwargs.pop("log_level").upper()
    exc_info = kwargs.pop("exc_info", False)

    logging.basicConfig(level=log_level_mapping.get(log_level, logging.INFO))

    for handler in logging.root.handlers:
        handler.setFormatter(LevelAwareLoggingFormatter())

    logging.info = partial(logging.info, exc_info=exc_info)
    logging.debug = partial(logging.debug, exc_info=exc_info)
    logging.error = partial(logging.error, exc_info=exc_info)
    logging.critical = partial(logging.critical, exc_info=exc_info)
    logging.warning = partial(logging.warning, exc_info=exc_info)

    from mehbar._internals import main as _main

    _main.entrypoint(**kwargs)
