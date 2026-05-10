from mehbar.widget import WidgetBase


class WidgetStatic(WidgetBase):
    UNIQUE = False

    TYPE = "static"

    def __init__(self, label_format: str):
        super().__init__(0, label_format)
        self.set_label(self.label_format)
