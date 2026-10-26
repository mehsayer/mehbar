import pytest

from mehbar.actions import ExecAction
from mehbar.exceptions import BarConfigError


def test_string_is_split_like_a_shell():
    assert ExecAction("notify-send 'two words'").args == ["notify-send", "two words"]


def test_list_is_used_as_is():
    assert ExecAction(["a", "b c"]).args == ["a", "b c"]


@pytest.mark.parametrize("args", ["", [], ["cmd", 5], 5, "'unclosed"])
def test_invalid_commands_are_config_errors(args):
    with pytest.raises(BarConfigError, match="'onclick' 1"):
        ExecAction(args, "'onclick' 1")


def test_failing_command_is_logged(caplog):
    ExecAction(["/nonexistent/command"]).run()
    assert "cannot execute" in caplog.text
