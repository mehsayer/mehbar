import os

import anyio
import pytest

from mehbar._widgets import _backlight
from mehbar._widgets._backlight import BacklightACPI


@pytest.fixture
def panel(tmp_path, monkeypatch):
    """A fake backlight device with 15 levels."""
    sys_bl = tmp_path / "backlight"
    device = sys_bl / "panel0"
    device.mkdir(parents=True)
    (device / "max_brightness").write_text("15\n")
    (device / "brightness").write_text("3\n")
    (tmp_path / "drm").mkdir()

    monkeypatch.setattr(BacklightACPI, "PATH_BL", str(sys_bl))
    monkeypatch.setattr(BacklightACPI, "PATH_DRM", str(tmp_path / "drm"))

    return device


def _raw(device) -> int:
    return int((device / "brightness").read_text())


async def _change(delta):
    async with BacklightACPI() as driver:
        return await driver.change_level(delta)


@pytest.mark.parametrize(("delta", "raw"), [(1, 4), (-1, 2), (20, 6), (-50, 0)])
def test_steps_by_at_least_one_level(panel, delta, raw):
    # 1% of 15 levels would round to no change
    level = anyio.run(_change, delta)

    assert _raw(panel) == raw
    assert level == pytest.approx(raw / 0.15)


def test_level_is_clamped(panel):
    (panel / "brightness").write_text("15\n")
    anyio.run(_change, 5)
    assert _raw(panel) == 15


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anyway")
def test_without_write_access_logind_is_used(panel, monkeypatch):
    calls = []
    monkeypatch.setattr(
        _backlight.LogindSession,
        "set_brightness",
        lambda self, device, level: calls.append((device, level)),
    )
    (panel / "brightness").chmod(0o444)

    anyio.run(_change, 1)

    assert calls == [("panel0", 4)]
    assert _raw(panel) == 3
