from mehbar.widgets import BarWidget

class BarWidgetStatic(BarWidget):
    def __init__(self, label_format: str):
        super().__init__(0, label_format)
        self.set_label(self.label_format)

    def update(self):
        pass
