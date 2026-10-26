"""The theme bundles in the package must be built from `examples/`."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
THEMES = ["_base", "tokyo-night"]


@pytest.mark.skipif(
    shutil.which("glib-compile-resources") is None,
    reason="needs glib-compile-resources",
)
@pytest.mark.parametrize("theme", THEMES)
def test_bundle_is_up_to_date(tmp_path, theme):
    built = tmp_path / f"{theme}.gresource"

    subprocess.run(
        [
            ROOT / "compile-gresources.sh",
            f"/org/mehbar/themes/{theme}",
            ROOT / "examples" / theme,
            built,
        ],
        check=True,
        capture_output=True,
        # without user tools, e.g. an svgo installed with npm
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    )

    packaged = ROOT / "src" / "mehbar" / "assets" / f"{theme}.gresource"

    assert built.read_bytes() == packaged.read_bytes(), (
        f"rebuild {packaged.relative_to(ROOT)} with compile-gresources.sh"
    )
