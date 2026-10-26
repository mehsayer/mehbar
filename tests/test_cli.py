import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def _mehbar(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "mehbar", *args],
        cwd=SRC,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def test_help():
    result = _mehbar("--help")
    assert result.returncode == 0
    assert "configuration directory path" in result.stdout


def test_missing_explicit_config_dir_is_an_error():
    result = _mehbar("-c", "/nonexistent/mehbar")
    assert result.returncode == 2
    assert "'/nonexistent/mehbar' is not an existing directory" in result.stderr


def test_unknown_option_is_an_error():
    result = _mehbar("--colour-scheme", "dark")
    assert result.returncode == 2
