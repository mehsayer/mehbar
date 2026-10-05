from mehbar.resource_manager import ResourceManager
from mehbar.widget import WidgetBase


class WidgetStatic(WidgetBase):
    """Shows the label as is, e.g. a separator."""

    UNIQUE = False
    STATIC = True

    TYPE = "static"

    def __init__(self, name: str, res_mgr: ResourceManager):
        super().__init__(name, res_mgr)
        self.apply_content(self.get_content())
