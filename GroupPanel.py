from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout
from PyQt5.QtGui import QIcon
import os

class GroupPanelWindow(QWidget):
    """
    Group panel window for creating groups.
    """
    def __init__(self, main_window, is_forecast=False):
        super().__init__()
        self.main_window = main_window
        self.is_forecast = is_forecast

        self.setWindowTitle("Create Group")
        self.setGeometry(100, 100, 300, 150)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "assets", "Logo_NBackground.png")
        self.setWindowIcon(QIcon(icon_path))

        layout = QVBoxLayout()

        self.name_label = QLabel("Name:", self)
        layout.addWidget(self.name_label)

        self.name_input = QLineEdit(self)
        layout.addWidget(self.name_input)

        self.accept_button = QPushButton("Confirm", self)
        self.accept_button.clicked.connect(self.accept)
        layout.addWidget(self.accept_button)

        self.cancel_button = QPushButton("Decline", self)
        self.cancel_button.clicked.connect(self.cancel)
        layout.addWidget(self.cancel_button)

        self.setLayout(layout)

    def accept(self):
        group_name = self.name_input.text().strip()
        if group_name:
            if self.is_forecast:
                self.main_window.create_forecast_group(group_name)
            else:
                self.main_window.create_group(group_name)
            self.close()
        else:
            self.main_window.console.append("Enter a valid name for the group.")

    def cancel(self):
        self.close()
