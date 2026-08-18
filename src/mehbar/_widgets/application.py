from gi.repository import Gio  # type: ignore

from mehbar.exceptions import BarConfigError
from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


class WidgetApplication(WidgetBase):
    TYPE = "application"
    STATIC = True
    UNIQUE = False

    APP_LIST: list[Gio.AppInfo] = []

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)

        self.app_id = self.cfg.get("app_id", "!")
        self.app_name = self.cfg.get("app_name", "!")

        self.files = None

        if (files_ := self.cfg.get("files")) is not None:
            if isinstance(files_, list):
                self.files = [Gio.File.new_for_path(path) for path in files_]

        if self.app_id is None and self.app_name is None:
            raise BarConfigError("application name or ID must be specififed")

        if not self.APP_LIST:
            self.APP_LIST = Gio.AppInfo.get_all()

        self.app = None

        for app_ in self.APP_LIST:
            if app_.get_id() == self.app_id or app_.get_name() == self.app_name:
                self.app = app_
                break

        self.set_label(self.content.label)

        icon = None
        if self.app is not None:
            if self.content.icon is not None:
                icon = self.content.icon
            else:
                if icon_names := self.app.get_icon().get_names():
                    icon = "icontheme://" + icon_names[0]
            self.set_icon(icon)
            self.onclick_call(1, self.app.launch, self.files)
