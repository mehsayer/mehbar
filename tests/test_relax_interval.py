import pytest

from mehbar.resource_manager import ResourceManager


@pytest.fixture
def res_mgr():
    # relax_interval() needs no display
    res_mgr = ResourceManager.__new__(ResourceManager)
    res_mgr._intervals = {}
    return res_mgr


def test_distinct_intervals_are_kept(res_mgr):
    assert res_mgr.relax_interval(1)[0] == 1
    assert res_mgr.relax_interval(10)[0] == 10
    assert res_mgr.relax_interval(60)[0] == 60


def test_equal_intervals_are_spread(res_mgr):
    relaxed = [res_mgr.relax_interval(3) for _ in range(6)]
    assert [interval for interval, _ in relaxed] == [3, 3.5, 4, 4.5, 5, 5.5]
    assert all(phase == 0 for _, phase in relaxed)


@pytest.mark.parametrize("interval", [0.2, 1, 2, 3, 5, 10, 30])
def test_deviation_is_capped(res_mgr, interval):
    """At most the interval itself (100%) and `INTERVAL_MAX_DEVIATION`."""
    limit = interval + min(interval, ResourceManager.INTERVAL_MAX_DEVIATION)

    for _ in range(20):
        relaxed, phase = res_mgr.relax_interval(interval)
        assert interval <= relaxed <= limit
        assert 0 <= phase < 1


def test_shared_intervals_get_distinct_phases(res_mgr):
    results = [res_mgr.relax_interval(3) for _ in range(10)]
    by_interval: dict[float, list[float]] = {}

    for interval, phase in results:
        by_interval.setdefault(interval, []).append(phase)

    # 3..5.5 s are used by six widgets, four more share the least used ones
    assert sorted(by_interval) == [3, 3.5, 4, 4.5, 5, 5.5]

    for phases in by_interval.values():
        assert len(set(phases)) == len(phases)


def test_relaxing_spreads_wakeups(res_mgr):
    """Ten widgets polling every 3 seconds do not all wake up together."""
    periods = []

    for _ in range(10):
        interval, phase = res_mgr.relax_interval(3)
        periods.append((interval, phase * interval))

    # count widgets waking up in the same 100 ms slots over a minute
    slots: dict[int, int] = {}

    for interval, phase in periods:
        t = phase
        while t < 60:
            slot = int(t * 10)
            slots[slot] = slots.get(slot, 0) + 1
            t += interval

    # all wake up at 0 s, the first time; then no more than 3 at once
    del slots[0]
    assert max(slots.values()) <= 3
