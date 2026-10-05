import argparse
import logging
import os
import sys
from pathlib import Path

from mehbar.tools import (
    ArgumentsHelpFormatter,
    ExceptionInfoFilter,
    LevelAwareLoggingFormatter,
    get_config_home,
)


def main() -> int:
    log_level_mapping = logging.getLevelNamesMapping()
    log_levels = [name.lower() for name in log_level_mapping.keys()]
    color_schemes = ["light", "dark", "system"]

    parser = argparse.ArgumentParser(
        prog="mehbar",
        formatter_class=ArgumentsHelpFormatter,
        description="Mehbar, a highly customizable status bar for Linux",
        epilog="Command line options override configuration file settings. \nCopyright (c) 2026, Mehsayer",
        suggest_on_error=True,
    )

    parser.add_argument(
        "-c",
        "--config-dir",
        dest="cfg_dir",
        type=Path,
        default=get_config_home(),
        help="configuration directory path",
    )

    parser.add_argument(
        "-L",
        "--log-level",
        choices=log_levels,
        dest="log_level",
        metavar="LEVEL",
        default="debug",
        help=f"log level, one of {', '.join([repr(level) for level in log_levels])}",
    )

    parser.add_argument(
        "-E",
        "--exception-info",
        action="store_true",
        dest="exc_info",
        help="print exception information to stdout",
    )

    parser.add_argument(
        "-s",
        "--color-scheme",
        choices=color_schemes,
        default=None,
        metavar="COLOR_SCHEME",
        dest="color_scheme",
        help=f"color scheme name, one of {', '.join([repr(cs) for cs in color_schemes])}",
    )
    parser.add_argument(
        "-t",
        "--theme",
        default=None,
        dest="theme",
        type=str,
        help="theme name",
    )
    args = parser.parse_args()

    if not args.cfg_dir.is_dir():
        parser.error(f"'{args.cfg_dir}' is not an existing directory")

    logging.basicConfig(level=log_level_mapping[args.log_level.upper()])

    for handler in logging.root.handlers:
        handler.setFormatter(LevelAwareLoggingFormatter())
        if args.exc_info:
            handler.addFilter(ExceptionInfoFilter())

    import mehbar._main

    return mehbar._main.entrypoint(
        cfg_dir=args.cfg_dir, theme=args.theme, color_scheme=args.color_scheme
    )


if __name__ == "__main__":
    # Do not let modules in the current working directory shadow dependencies
    if sys.path[0] in ("", os.getcwd()):
        sys.path.pop(0)

    sys.exit(main())
