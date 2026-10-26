from types import SimpleNamespace

import anyio
import pytest

pytest.importorskip("i3ipc")

pytestmark = pytest.mark.display


def _workspace(num, focused=False, urgent=False):
    return SimpleNamespace(
        name=str(num), num=num, focused=focused, visible=focused, urgent=urgent
    )


class _Connection:
    def __init__(self, workspaces):
        self.workspaces = workspaces

    async def get_workspaces(self):
        return self.workspaces


def _shown(make_res_mgr, workspaces, options=""):
    from mehbar._widgets.i3_workspaces import WidgetI3Workspaces

    res_mgr = make_res_mgr(f'[start.ws]\ntype = "i3_workspaces"\n{options}')
    widget = WidgetI3Workspaces("ws", res_mgr)

    submitted = []
    widget._updater.submit = submitted.append
    widget._i3_conn = _Connection(workspaces)

    anyio.run(widget._refresh)

    return [state.name for state in submitted[-1]]


def test_workspaces_are_ordered_by_number(make_res_mgr):
    workspaces = [_workspace(3), _workspace(1, focused=True), _workspace(2)]
    assert _shown(make_res_mgr, workspaces) == ["1", "2", "3"]


def test_focused_and_urgent_workspaces_are_never_cut(make_res_mgr):
    workspaces = [_workspace(n) for n in range(1, 10)]
    workspaces[8] = _workspace(9, focused=True)
    workspaces[6] = _workspace(7, urgent=True)

    shown = _shown(make_res_mgr, workspaces, "max_workspaces = 4")

    assert shown == ["1", "2", "7", "9"]
