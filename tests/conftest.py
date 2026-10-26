import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")

import pytest  # noqa: E402

# importing Gtk initializes it, opening the default display
from gi.repository import Gdk, Gtk  # noqa: E402, F401


def pytest_collection_modifyitems(config, items):
    if Gdk.Display.get_default() is not None:
        return

    skip = pytest.mark.skip(reason="needs a Wayland display")

    for item in items:
        if "display" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def make_res_mgr(tmp_path):
    """Returns a function creating a `ResourceManager` for a configuration
    directory containing `config_toml`."""
    from mehbar.resource_manager import ResourceManager

    def _make(config_toml: str = "", theme=None, color_scheme=None):
        (tmp_path / "config.toml").write_text(config_toml)
        return ResourceManager(tmp_path, theme, color_scheme)

    return _make
