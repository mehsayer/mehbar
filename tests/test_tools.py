import math

import pytest

from mehbar import tools
from mehbar.exceptions import BarConfigError
from mehbar.tools import (
    FormattableTimeDelta,
    OptionalFormatter,
    describe_exception,
    next_prime,
    overlay_dict_r,
    split_format,
)


def test_next_prime():
    assert next_prime(3000) == 3001
    assert next_prime(3001) == 3001
    assert next_prime(0) == 2


def test_overlay_dict_r_merges_nested_values():
    bottom = {"a": 1, "b": {"c": 2, "d": 3}}
    overlay_dict_r(bottom, {"b": {"c": 20}, "e": 5})
    assert bottom == {"a": 1, "b": {"c": 20, "d": 3}, "e": 5}


def test_overlay_dict_r_limits_depth():
    deep = current = {}
    for _ in range(20):
        current["x"] = {}
        current = current["x"]

    with pytest.raises(ValueError):
        overlay_dict_r({}, deep)


@pytest.mark.parametrize(
    ("seconds", "fmt", "expected"),
    [
        (3725, "%h:%M:%S", "1:02:05"),
        (3725, "%b min", "62 min"),
        (90061, "%dd %H:%M", "1d 01:01"),
        (5, "100%%", "100%"),
        (5, "%q", "%q"),
        # negative durations, e.g. clock skew, are shown as zero
        (-5, "%h:%M:%S", "0:00:00"),
    ],
)
def test_formattable_time_delta(seconds, fmt, expected):
    assert FormattableTimeDelta(seconds).strftime(fmt) == expected


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_formattable_time_delta_rejects_non_finite(value):
    with pytest.raises(ValueError):
        FormattableTimeDelta(value)


def test_optional_formatter_skips_missing_and_none_fields():
    fmt = OptionalFormatter()
    assert fmt.vformat("{a}-{b}-{c}", (), {"a": 1, "c": None}) == "1--"


def test_optional_formatter_falls_back_to_str_for_bad_specs():
    fmt = OptionalFormatter()
    assert fmt.vformat("{a:.1f}", (), {"a": "text"}) == "text"
    assert fmt.vformat("{a:.1f}", (), {"a": 1.25}) == "1.2"


def test_optional_formatter_accepts_sensor_names():
    fmt = OptionalFormatter()
    fields = {"k10temp/Tctl": 40, "nvme/Composite#2": 36}
    assert fmt.vformat("{k10temp/Tctl} {nvme/Composite#2}", (), fields) == "40 36"
    assert fmt.get_fields("{a} and {b:>3}") == ["a", "b"]


def test_split_format():
    assert split_format("cpu {ramp} {percent}%", "ramp") == (
        "cpu ",
        "{ramp}",
        " {percent}%",
    )
    assert split_format("{percent}", "ramp") is None
    assert split_format(None, "ramp") is None


def test_describe_exception():
    assert describe_exception(ValueError("bad")) == "ValueError: bad"
    assert describe_exception(TimeoutError()) == "TimeoutError"
    assert describe_exception(BarConfigError("wrong")) == "wrong"

    group = ExceptionGroup("group", [KeyError("k"), ValueError("v")])
    assert describe_exception(group) == "KeyError: 'k'"


def test_child_environ_drops_own_variables(monkeypatch):
    monkeypatch.setattr(tools, "OWN_ENVIRON", set())
    assert tools.child_environ() is None

    monkeypatch.setenv("MEHBAR_TEST_VAR", "1")
    monkeypatch.setattr(tools, "OWN_ENVIRON", {"MEHBAR_TEST_VAR"})
    env = tools.child_environ()
    assert env is not None and "MEHBAR_TEST_VAR" not in env and "PATH" in env
