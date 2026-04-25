from mehbar.widgets import Widget


class WidgetStatic(Widget):
    def __init__(self, label_format: str):
        super().__init__(0, label_format)
        self.set_label(self.label_format)
