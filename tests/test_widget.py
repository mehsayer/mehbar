import logging
import re

import anyio
import pytest
from gi.repository import Gtk

from mehbar.exceptions import BarConfigError
from mehbar.widget import IconPosition, RewriteMixin, WidgetContent


def test_parse_plain_text():
    content = WidgetContent.parse("{percent}%")
    assert content.label == "{percent}%"
    assert content.icon is None
    assert content.icon_position == IconPosition.NONE


def test_parse_specification():
    content = WidgetContent.parse(
        "[icon=cpu>; classes:icon=warning, label=dim bold; tooltip=load {percent};]{percent}%"
    )
    assert content.label == "{percent}%"
    assert content.icon == "cpu"
    assert content.icon_position == IconPosition.END
    assert content.icon_classes == {"warning"}
    assert content.label_classes == {"dim", "bold"}
    assert content.tooltip_text == "load {percent}"


@pytest.mark.parametrize(
    "text",
    [
        "[icon=a;][icon=b;]",
        "[icon=a; icon=b;]",
        "[classes=1bad;]",
        "[tooltip= ;]",
    ],
)
def test_parse_errors(text):
    with pytest.raises(BarConfigError):
        WidgetContent.parse(text)


def test_brackets_without_known_fields_are_text():
    assert WidgetContent.parse("[1/4]").label == "[1/4]"


class _Rewriter:
    _rewrite = RewriteMixin._rewrite

    def __init__(self, rules):
        self._rewrite_rules = [(re.compile(p), r) for p, r in rules.items()]


@pytest.mark.parametrize(
    ("rules", "text", "expected"),
    [
        ({"Eng.+": "EN"}, "English (US)", "EN"),
        ({"Eng\\w+": "EN"}, "English (US)", "EN (US)"),
        ({"Ukr\\w+": "UA", "Eng\\w+": "EN"}, "Ukrainian", "UA"),
        # only the match at the start is replaced
        ({"a": "b"}, "abacus", "bbacus"),
        # expressions match at the start only
        ({"US": "x"}, "English (US)", "English (US)"),
    ],
)
def test_rewrite(rules, text, expected):
    assert _Rewriter(rules)._rewrite(text) == expected


# Widgets, these need a display


@pytest.fixture
def static_widget(make_res_mgr):
    from mehbar._widgets.static import WidgetStatic

    def _make(options: str = ""):
        res_mgr = make_res_mgr(f'[start.w]\ntype = "static"\nlabel = "x"\n{options}')
        widget = WidgetStatic("w", res_mgr)
        parent = Gtk.Box()
        parent.append(widget)
        return widget

    return _make


@pytest.mark.display
def test_failing_callback_disables_the_widget(static_widget, caplog):
    widget = static_widget()
    calls = []

    def _callback():
        calls.append(1)
        raise ValueError("boom")

    guarded = widget.guard(_callback)

    with caplog.at_level(logging.ERROR):
        assert guarded() is None

    assert widget.get_parent() is None
    assert "disabling widget 'w': ValueError: boom" in caplog.text

    # disabled widgets do not run callbacks, timeouts are removed
    assert guarded() is None
    assert calls == [1]


@pytest.mark.display
def test_empty_scroll_command_is_allowed(static_widget):
    static_widget('onscroll = ["", "true"]')


@pytest.mark.display
def test_invalid_click_command_is_a_config_error(static_widget):
    with pytest.raises(BarConfigError, match="'onclick' 1"):
        static_widget('onclick = [["true", 5]]')


@pytest.mark.display
def test_polling_widget_without_interval_is_reported(make_res_mgr, caplog):
    from mehbar._widgets.memory import WidgetMemoryUsage

    res_mgr = make_res_mgr('[start.m]\ntype = "memory_usage"')
    WidgetMemoryUsage("m", res_mgr)

    assert "widget 'm' has no 'interval'" in caplog.text


@pytest.mark.display
def test_widgets_sharing_an_interval_wake_up_apart(make_res_mgr):
    from mehbar._widgets.memory import WidgetMemoryUsage

    cfg = "".join(
        f'[start.m{i}]\ntype = "memory_usage"\ninterval = 0.2\nrelax_interval = true\n'
        for i in range(3)
    )
    res_mgr = make_res_mgr(cfg)
    widgets = [WidgetMemoryUsage(f"m{i}", res_mgr) for i in range(3)]

    async def _first_deadlines():
        deadlines = []
        for widget in widgets:
            assert await widget.sleep_interval()
            deadlines.append(widget._deadline - anyio.current_time())
        return deadlines

    deadlines = anyio.run(_first_deadlines)

    # 0.2 s may be relaxed to 0.4 s at most, which is not 0.5 s apart: the
    # widgets share the interval, but wake up at different times
    assert [w.interval for w in widgets] == [0.2, 0.2, 0.2]
    assert len({round(d, 3) for d in deadlines}) == 3
