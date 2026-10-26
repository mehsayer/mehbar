import pytest
from gi.repository import Gtk

pytestmark = pytest.mark.display


def _widget(make_res_mgr, options="", widget_type="static", bar=""):
    from mehbar._widgets import get_widget_class

    res_mgr = make_res_mgr(
        f'[bar]\n{bar}\n[start.w]\ntype = "{widget_type}"\n{options}'
    )
    widget = get_widget_class(widget_type)("w", res_mgr)
    Gtk.Box().append(widget)
    return widget


def _texts(widget: Gtk.Widget) -> list[str]:
    texts = [widget.get_label()] if isinstance(widget, Gtk.Label) else []
    child = widget.get_first_child()
    while child is not None:
        texts += _texts(child)
        child = child.get_next_sibling()
    return texts


def test_tooltip_is_formatted_like_the_label(make_res_mgr):
    widget = _widget(
        make_res_mgr,
        'interval = 3\nlabel = "{percent}%"\ntooltip = "[icon=cpu;]Load {percent}%"',
        "cpu_usage",
    )
    widget.apply_content(widget.get_content(42, percent=42))

    tooltip = widget.build_tooltip()

    assert _texts(tooltip) == ["Load 42%"]
    assert isinstance(tooltip.get_first_child(), Gtk.Image)


def test_empty_tooltip_disables_the_default_one(make_res_mgr):
    widget = _widget(make_res_mgr, 'interval = 3\ntooltip = ""', "cpu_usage")
    assert widget.build_tooltip() is None


def test_datetime_tooltip_is_a_calendar(make_res_mgr):
    widget = _widget(make_res_mgr, "interval = 60", "datetime")
    assert isinstance(widget.build_tooltip(), Gtk.Calendar)


def test_click_trigger_shows_a_popover(make_res_mgr):
    widget = _widget(make_res_mgr, 'tooltip = "hello"', bar='tooltip_trigger = "click"')
    widget._setup_tooltip()

    # the popup is tested in a running bar, it needs a mapped window
    assert widget._popover is not None and not widget.get_has_tooltip()
    assert _texts(widget.build_tooltip()) == ["hello"]


def test_click_trigger_falls_back_to_hover(make_res_mgr):
    # the left button runs the command
    widget = _widget(
        make_res_mgr,
        'tooltip = "hello"\ntooltip_trigger = "click"\nonclick = ["true"]',
    )
    widget._setup_tooltip()

    assert widget._popover is None and widget.get_has_tooltip()


def test_invalid_trigger_is_a_config_error(make_res_mgr):
    from mehbar.exceptions import BarConfigError

    with pytest.raises(BarConfigError, match="tooltip_trigger"):
        _widget(make_res_mgr, 'tooltip_trigger = "never"')


def test_idle_inhibitor_without_window_stays_off(make_res_mgr):
    widget = _widget(make_res_mgr, "", "idle_inhibitor")

    widget.toggle()

    assert not widget.active
    assert _texts(widget.build_tooltip()) == ["Idle inhibitor: off"]
