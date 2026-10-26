import logging

import pytest

from mehbar import resource_manager
from mehbar.resource_manager import ResourceManager

pytestmark = pytest.mark.display


def test_base_widgets_are_used_without_user_widgets(make_res_mgr):
    res_mgr = make_res_mgr("[bar]\nheight = 30\n")

    assert res_mgr.bar_cfg["height"] == 30
    assert any(
        res_mgr.cfg.get(section) for section in ResourceManager.WIDGET_CFG_SECTIONS
    )


def test_user_widgets_replace_base_widgets(make_res_mgr):
    res_mgr = make_res_mgr('[end.clock]\ntype = "datetime"\ninterval = 60\n')

    sections = {s: res_mgr.cfg.get(s, {}) for s in ResourceManager.WIDGET_CFG_SECTIONS}
    assert sections == {
        "start": {},
        "center": {},
        "end": {"clock": res_mgr.get_cfg_for_name("clock")},
    }


def test_user_options_override_the_theme(make_res_mgr):
    res_mgr = make_res_mgr('[bar]\ntheme = "tokyo-night"\nheight = 31\n')

    assert res_mgr.theme == "tokyo-night"
    assert res_mgr.bar_cfg["height"] == 31


def test_unknown_theme_is_not_active(make_res_mgr, caplog):
    with caplog.at_level(logging.ERROR):
        res_mgr = make_res_mgr(theme="no-such-theme")

    assert "theme 'no-such-theme' not found" in caplog.text
    assert res_mgr.theme == ResourceManager.BASE_THEME


@pytest.mark.parametrize(
    ("argument", "expected"), [(None, "dark"), ("light", "light"), ("system", "light")]
)
def test_command_line_color_scheme_wins(make_res_mgr, monkeypatch, argument, expected):
    # 'system' follows the desktop, which prefers nothing here
    monkeypatch.setattr(resource_manager, "detect_color_scheme", lambda: None)

    res_mgr = make_res_mgr('[bar]\ncolor_scheme = "dark"\n', color_scheme=argument)

    assert res_mgr.color_scheme == expected


def test_css_layers_are_in_order_of_precedence(make_res_mgr, tmp_path):
    (tmp_path / "style.css").write_text("/* user */")
    res_mgr = make_res_mgr('[bar]\ntheme = "tokyo-night"\n')

    labels = [label for label, _ in res_mgr.get_css()]

    assert labels[0].startswith("resource:///org/mehbar/themes/_base/")
    assert "tokyo-night" in labels[1]
    assert labels[-1] == f"{tmp_path}/style.css"
