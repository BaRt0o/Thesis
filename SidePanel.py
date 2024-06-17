from PyQt5.QtWidgets import QWidget, QComboBox, QPushButton, QLineEdit, QLabel, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import os

class SidePanelWindow(QWidget):
    """
    Side panel window for settings.
    """
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        self.setWindowTitle("Settings")
        self.setGeometry(100, 100, 300, 800)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "assets", "Logo_NBackground.png")
        self.setWindowIcon(QIcon(icon_path))

        self.init_model_settings_ui()
        self.init_plot_settings_ui()
        self.init_correction_settings_ui()  

        self.show_model_settings()

    def init_model_settings_ui(self):
        self.left_arrow_button = QPushButton("←", self)
        self.left_arrow_button.setGeometry(10, 10, 30, 30)
        self.left_arrow_button.clicked.connect(self.show_plot_settings)

        self.right_arrow_button = QPushButton("→", self)
        self.right_arrow_button.setGeometry(260, 10, 30, 30)
        self.right_arrow_button.clicked.connect(self.show_plot_settings)

        self.title_section1 = QLabel("📈 Model Settings", self)
        self.title_section1.setGeometry(60, 10, 180, 30)
        self.title_section1.setAlignment(Qt.AlignCenter)

        self.model_combo = QComboBox(self)
        self.model_combo.addItems(["SARIMAX", "ARIMAX", "ARIMA"])
        self.model_combo.currentTextChanged.connect(self.update_model_parameters)
        self.model_combo.setGeometry(10, 50, 280, 30)

        self.p_range_label = QLabel("p_range :", self)
        self.p_range_label.setGeometry(10, 90, 100, 30)
        self.p_range_input = QLineEdit(self)
        self.p_range_input.setText("0,2")
        self.p_range_input.setGeometry(140, 90, 150, 30)

        self.d_range_label = QLabel("d_range :", self)
        self.d_range_label.setGeometry(10, 130, 100, 30)
        self.d_range_input = QLineEdit(self)
        self.d_range_input.setText("0,1")
        self.d_range_input.setGeometry(140, 130, 150, 30)

        self.q_range_label = QLabel("q_range :", self)
        self.q_range_label.setGeometry(10, 170, 100, 30)
        self.q_range_input = QLineEdit(self)
        self.q_range_input.setText("0,2")
        self.q_range_input.setGeometry(140, 170, 150, 30)

        self.seasonal_period_label = QLabel("Seasonality :", self)
        self.seasonal_period_label.setGeometry(10, 210, 120, 30)
        self.seasonal_period_input = QLineEdit(self)
        self.seasonal_period_input.setText("12")
        self.seasonal_period_input.setGeometry(140, 210, 150, 30)

        self.forecast_until_label = QLabel("Forecast year:", self)
        self.forecast_until_label.setGeometry(10, 250, 120, 30)
        self.forecast_until_input = QLineEdit(self)
        self.forecast_until_input.setText("2100")
        self.forecast_until_input.setGeometry(140, 250, 150, 30)

        self.replace_negative_forecast_checkbox = QCheckBox("0 values", self)
        self.replace_negative_forecast_checkbox.setGeometry(10, 290, 280, 30)

        self.show_confidence_interval_checkbox = QCheckBox("Show Confidence Interval", self)
        self.show_confidence_interval_checkbox.setGeometry(10, 330, 280, 30)

        self.apply_button = QPushButton("Apply Settings", self)
        self.apply_button.clicked.connect(self.apply_model)
        self.apply_button.setGeometry(10, 370, 280, 30)

    def init_plot_settings_ui(self):
        self.left_arrow_button2 = QPushButton("←", self)
        self.left_arrow_button2.setGeometry(10, 10, 30, 30)
        self.left_arrow_button2.clicked.connect(self.show_model_settings)
        self.left_arrow_button2.setVisible(False)

        self.right_arrow_button2 = QPushButton("→", self)
        self.right_arrow_button2.setGeometry(260, 10, 30, 30)
        self.right_arrow_button2.clicked.connect(self.show_model_settings)
        self.right_arrow_button2.setVisible(False)

        self.title_section2 = QLabel("📊 Plot Settings", self)
        self.title_section2.setGeometry(60, 10, 180, 30)
        self.title_section2.setAlignment(Qt.AlignCenter)
        self.title_section2.setVisible(False)

        self.x_axis_label = QLabel("X-axis :", self)
        self.x_axis_label.setGeometry(10, 50, 130, 30)
        self.x_axis_label.setVisible(False)
        self.x_axis_input = QLineEdit(self)
        self.x_axis_input.setText("0,2100")
        self.x_axis_input.setGeometry(150, 50, 140, 30)
        self.x_axis_input.setVisible(False)

        self.y_axis_label = QLabel("Y-axis :", self)
        self.y_axis_label.setGeometry(10, 90, 130, 30)
        self.y_axis_label.setVisible(False)
        self.y_axis_input = QLineEdit(self)
        self.y_axis_input.setText("0,100")
        self.y_axis_input.setGeometry(150, 90, 140, 30)
        self.y_axis_input.setVisible(False)

        self.title_label = QLabel("Chart Title:", self)
        self.title_label.setGeometry(10, 130, 130, 30)
        self.title_label.setVisible(False)
        self.title_input = QLineEdit(self)
        self.title_input.setText("Chart Title")
        self.title_input.setGeometry(150, 130, 140, 30)
        self.title_input.setVisible(False)

        self.legend_size_label = QLabel("Legend Size:", self)
        self.legend_size_label.setGeometry(10, 170, 130, 30)
        self.legend_size_label.setVisible(False)
        self.legend_size_input = QLineEdit(self)
        self.legend_size_input.setText("10")
        self.legend_size_input.setGeometry(150, 170, 140, 30)
        self.legend_size_input.setVisible(False)

        self.ylabel_label = QLabel("Y-axis Label:", self)
        self.ylabel_label.setGeometry(10, 210, 130, 30)
        self.ylabel_label.setVisible(False)
        self.ylabel_input = QLineEdit(self)
        self.ylabel_input.setText("Y-axis Label")
        self.ylabel_input.setGeometry(150, 210, 140, 30)
        self.ylabel_input.setVisible(False)

        self.xlabel_label = QLabel("X-axis Label:", self)
        self.xlabel_label.setGeometry(10, 250, 130, 30)
        self.xlabel_label.setVisible(False)
        self.xlabel_input = QLineEdit(self)
        self.xlabel_input.setText("X-axis Label")
        self.xlabel_input.setGeometry(150, 250, 140, 30)
        self.xlabel_input.setVisible(False)

        self.ylabel_size_label = QLabel("Y-axis Label Size:", self)
        self.ylabel_size_label.setGeometry(10, 290, 130, 30)
        self.ylabel_size_label.setVisible(False)
        self.ylabel_size_input = QLineEdit(self)
        self.ylabel_size_input.setText("14")
        self.ylabel_size_input.setGeometry(150, 290, 140, 30)
        self.ylabel_size_input.setVisible(False)

        self.xlabel_size_label = QLabel("X-axis Label Size:", self)
        self.xlabel_size_label.setGeometry(10, 330, 130, 30)
        self.xlabel_size_label.setVisible(False)
        self.xlabel_size_input = QLineEdit(self)
        self.xlabel_size_input.setText("14")
        self.xlabel_size_input.setGeometry(150, 330, 140, 30)
        self.xlabel_size_input.setVisible(False)

        self.apply_plot_button = QPushButton("Apply Plot Settings", self)
        self.apply_plot_button.clicked.connect(self.apply_plot_settings)
        self.apply_plot_button.setGeometry(10, 370, 280, 30)
        self.apply_plot_button.setVisible(False)

    def show_model_settings(self):
        self.title_section1.setVisible(True)
        self.left_arrow_button.setVisible(True)
        self.right_arrow_button.setVisible(True)
        self.model_combo.setVisible(True)
        self.p_range_label.setVisible(True)
        self.p_range_input.setVisible(True)
        self.d_range_label.setVisible(True)
        self.d_range_input.setVisible(True)
        self.q_range_label.setVisible(True)
        self.q_range_input.setVisible(True)
        self.seasonal_period_label.setVisible(self.model_combo.currentText() == "SARIMAX")
        self.seasonal_period_input.setVisible(self.model_combo.currentText() == "SARIMAX")
        self.forecast_until_label.setVisible(True)
        self.forecast_until_input.setVisible(True)
        self.replace_negative_forecast_checkbox.setVisible(True)
        self.show_confidence_interval_checkbox.setVisible(True)
        self.target_year_label.setVisible(True)
        self.target_year_input.setVisible(True)
        self.target_value_label.setVisible(True)
        self.target_value_input.setVisible(True)
        self.continuous_correction_checkbox.setVisible(True)
        self.short_correction_checkbox.setVisible(True)
        self.start_correction_checkbox.setVisible(True)
        self.apply_correction_button.setVisible(True)
        self.apply_button.setVisible(True)

        self.title_section2.setVisible(False)
        self.left_arrow_button2.setVisible(False)
        self.right_arrow_button2.setVisible(False)
        self.x_axis_label.setVisible(False)
        self.x_axis_input.setVisible(False)
        self.y_axis_label.setVisible(False)
        self.y_axis_input.setVisible(False)
        self.title_label.setVisible(False)
        self.title_input.setVisible(False)
        self.legend_size_label.setVisible(False)
        self.legend_size_input.setVisible(False)
        self.ylabel_label.setVisible(False)
        self.ylabel_input.setVisible(False)
        self.xlabel_label.setVisible(False)
        self.xlabel_input.setVisible(False)
        self.ylabel_size_label.setVisible(False)
        self.ylabel_size_input.setVisible(False)
        self.xlabel_size_label.setVisible(False)
        self.xlabel_size_input.setVisible(False)
        
        self.apply_plot_button.setVisible(False)

    def show_plot_settings(self):
        self.title_section1.setVisible(False)
        self.left_arrow_button.setVisible(False)
        self.right_arrow_button.setVisible(False)
        self.model_combo.setVisible(False)
        self.p_range_label.setVisible(False)
        self.p_range_input.setVisible(False)
        self.d_range_label.setVisible(False)
        self.d_range_input.setVisible(False)
        self.q_range_label.setVisible(False)
        self.q_range_input.setVisible(False)
        self.seasonal_period_label.setVisible(False)
        self.seasonal_period_input.setVisible(False)
        self.forecast_until_label.setVisible(False)
        self.forecast_until_input.setVisible(False)
        self.replace_negative_forecast_checkbox.setVisible(False)
        self.show_confidence_interval_checkbox.setVisible(False)
        self.target_year_label.setVisible(False)
        self.target_year_input.setVisible(False)
        self.target_value_label.setVisible(False)
        self.target_value_input.setVisible(False)
        self.continuous_correction_checkbox.setVisible(False)
        self.short_correction_checkbox.setVisible(False)
        self.start_correction_checkbox.setVisible(False)
        self.apply_correction_button.setVisible(False)
        self.apply_button.setVisible(False)

        self.title_section2.setVisible(True)
        self.left_arrow_button2.setVisible(True)
        self.right_arrow_button2.setVisible(True)
        self.x_axis_label.setVisible(True)
        self.x_axis_input.setVisible(True)
        self.y_axis_label.setVisible(True)
        self.y_axis_input.setVisible(True)
        self.title_label.setVisible(True)
        self.title_input.setVisible(True)
        self.legend_size_label.setVisible(True)
        self.legend_size_input.setVisible(True)
        self.ylabel_label.setVisible(True)
        self.ylabel_input.setVisible(True)
        self.xlabel_label.setVisible(True)
        self.xlabel_input.setVisible(True)
        self.ylabel_size_label.setVisible(True)
        self.ylabel_size_input.setVisible(True)
        self.xlabel_size_label.setVisible(True)
        self.xlabel_size_input.setVisible(True)
        self.apply_plot_button.setVisible(True)

    def update_model_parameters(self, model_name):
        is_sarimax = model_name == "SARIMAX"
        is_arimax = model_name == "ARIMAX"
        is_arima = model_name == "ARIMA"
        self.p_range_label.setVisible(is_sarimax or is_arimax or is_arima)
        self.p_range_input.setVisible(is_sarimax or is_arimax or is_arima)
        self.d_range_label.setVisible(is_sarimax or is_arimax or is_arima)
        self.d_range_input.setVisible(is_sarimax or is_arimax or is_arima)
        self.q_range_label.setVisible(is_sarimax or is_arimax or is_arima)
        self.q_range_input.setVisible(is_sarimax or is_arimax or is_arima)
        self.seasonal_period_label.setVisible(is_sarimax)
        self.seasonal_period_input.setVisible(is_sarimax)
        self.forecast_until_label.setVisible(is_sarimax or is_arimax or is_arima)
        self.forecast_until_input.setVisible(is_sarimax or is_arimax or is_arima)
        self.replace_negative_forecast_checkbox.setVisible(is_sarimax or is_arimax or is_arima)

    def apply_model(self):
        model_name = self.model_combo.currentText()
        if model_name == "SARIMAX":
            self.apply_sarimax()
        elif model_name == "ARIMAX":
            self.apply_arimax()
        elif model_name == "ARIMA":
            self.apply_arima()

    def apply_sarimax(self):
        p_range = self.get_range(self.p_range_input.text(), [0, 2])
        d_range = self.get_range(self.d_range_input.text(), [0, 1])
        q_range = self.get_range(self.q_range_input.text(), [0, 2])
        seasonal_period = int(self.seasonal_period_input.text()) if self.seasonal_period_input.text() else 11

        forecast_until_year = int(self.forecast_until_input.text()) if self.forecast_until_input.text() else 2100
        self.main_window.forecast_until_year = forecast_until_year

        self.main_window.replace_negative_forecast = self.replace_negative_forecast_checkbox.isChecked()

        self.main_window.run_sarimax(p_range, d_range, q_range, seasonal_period)

    def apply_arimax(self):
        p_range = self.get_range(self.p_range_input.text(), [0, 2])
        d_range = self.get_range(self.d_range_input.text(), [0, 1])
        q_range = self.get_range(self.q_range_input.text(), [0, 2])

        forecast_until_year = int(self.forecast_until_input.text()) if self.forecast_until_input.text() else 2100
        self.main_window.forecast_until_year = forecast_until_year

        self.main_window.replace_negative_forecast = self.replace_negative_forecast_checkbox.isChecked()

        self.main_window.run_arimax(p_range, d_range, q_range)

    def apply_arima(self):
        p_range = self.get_range(self.p_range_input.text(), [0, 2])
        d_range = self.get_range(self.d_range_input.text(), [0, 1])
        q_range = self.get_range(self.q_range_input.text(), [0, 2])

        forecast_until_year = int(self.forecast_until_input.text()) if self.forecast_until_input.text() else 2100
        self.main_window.forecast_until_year = forecast_until_year

        self.main_window.replace_negative_forecast = self.replace_negative_forecast_checkbox.isChecked()

        self.main_window.run_arima(p_range, d_range, q_range)

    def get_range(self, text, default):
        if text:
            try:
                values = list(map(int, text.split(',')))
                if len(values) == 2:
                    return values
                else:
                    return default
            except ValueError:
                return default
        else:
            return default

    def apply_plot_settings(self):
        x_range = self.get_range(self.x_axis_input.text(), None)
        y_range = self.get_range(self.y_axis_input.text(), None)
        title = self.title_input.text()
        legend_size = int(self.legend_size_input.text()) if self.legend_size_input.text() else None
        xlabel = self.xlabel_input.text()
        ylabel = self.ylabel_input.text()
        xlabel_size = int(self.xlabel_size_input.text()) if self.xlabel_size_input.text() else None
        ylabel_size = int(self.ylabel_size_input.text()) if self.ylabel_size_input.text() else None

        self.main_window.apply_plot_settings(x_range, y_range, title, legend_size, xlabel, ylabel, xlabel_size, ylabel_size)

    def init_correction_settings_ui(self):
        self.target_year_label = QLabel("Target Year:", self)
        self.target_year_label.setGeometry(10, 410, 120, 30)
        self.target_year_input = QLineEdit(self)
        self.target_year_input.setText("")
        self.target_year_input.setGeometry(140, 410, 150, 30)

        self.target_value_label = QLabel("Target Value:", self)
        self.target_value_label.setGeometry(10, 450, 120, 30)
        self.target_value_input = QLineEdit(self)
        self.target_value_input.setText("")
        self.target_value_input.setGeometry(140, 450, 150, 30)

        self.continuous_correction_checkbox = QCheckBox("Continuous Correction", self)
        self.continuous_correction_checkbox.setGeometry(10, 490, 280, 30)

        self.short_correction_checkbox = QCheckBox("Short Correction", self)
        self.short_correction_checkbox.setGeometry(10, 530, 280, 30)

        self.start_correction_checkbox = QCheckBox("Start Correction", self)
        self.start_correction_checkbox.setGeometry(10, 570, 280, 30)

        self.apply_correction_button = QPushButton("Apply Correction", self)
        self.apply_correction_button.clicked.connect(self.main_window.apply_forecast_corrections)
        self.apply_correction_button.setGeometry(10, 610, 280, 30)

