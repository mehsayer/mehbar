import logging
from dataclasses import replace

from gi.repository import Gio, GLib  # type: ignore

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


class WidgetApplication(WidgetBase):
    """Launches a desktop application on click. Label fields: `name`, `id`.
    Shows the application icon, unless the label specifies one."""

    TYPE = "application"
    STATIC = True
    UNIQUE = False

    # for applications without an icon
    FALLBACK_ICON = "icontheme://application-x-executable"

    _app_list: list[Gio.AppInfo] | None = None

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        app_id = self.cfg.get("app_id")
        app_name = self.cfg.get("app_name")

        if app_id is None and app_name is None:
            raise BarConfigError("application name or ID must be specified")

        files = self.cfg.get("files") or []

        if not isinstance(files, list):
            raise BarConfigError("'files' must be a list of paths")

        self.files = [Gio.File.new_for_path(str(path)) for path in files]

        if (app := self._find_app(app_id, app_name)) is None:
            raise BarConfigError(f"application '{app_id or app_name}' not found")

        self.app = app

        content = self.get_content(name=app.get_display_name(), id=app.get_id())

        if content.icon is None:
            content = replace(content, icon=self._get_app_icon() or self.FALLBACK_ICON)

        self.apply_content(content)
        self.onclick_call(1, self._launch)

    @classmethod
    def _find_app(cls, app_id: str | None, app_name: str | None) -> Gio.AppInfo | None:
        if app_id is not None:
            try:
                if (app := Gio.DesktopAppInfo.new(app_id)) is not None:
                    return app
            except TypeError:  # constructor returned NULL
                pass

        if cls._app_list is None:
            WidgetApplication._app_list = Gio.AppInfo.get_all()

        for app in cls._app_list or []:
            if app.get_id() == app_id or app.get_name() == app_name:
                return app

        return None

    def _get_app_icon(self) -> str | None:
        gicon = self.app.get_icon()

        if isinstance(gicon, Gio.ThemedIcon) and (icon_names := gicon.get_names()):
            return "icontheme://" + icon_names[0]

        if isinstance(gicon, Gio.FileIcon):
            return gicon.get_file().get_uri()

        return None

    def _launch(self):
        context = self.get_display().get_app_launch_context()

        try:
            self.app.launch(self.files or None, context)
        except GLib.Error as ex:
            logging.error("cannot launch '%s': %s", self.app.get_id(), ex.message)
