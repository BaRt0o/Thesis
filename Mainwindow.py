import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QComboBox, QTextEdit, QFileDialog, QLabel, QSpinBox, QListWidget, QListWidgetItem, QLineEdit)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from Adf_test import perform_adf_test_with_differencing
from Sarimax import optimize_sarimax_models, forecast_future as forecast_future_sarimax
from Arimax import optimize_arimax_models, forecast_future as forecast_future_arimax
from Arima import optimize_arima_models, forecast_future as forecast_future_arima
from Plotting import plot_data, plot_data_stacked_bar
from SidePanel import SidePanelWindow
from GroupPanel import GroupPanelWindow

class MainWindow(QMainWindow):
    """
    Main application window for forecasting.
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Forecasting")
        self.setGeometry(100, 100, 940, 880)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "assets", "Logo_NBackground.png")
        self.setWindowIcon(QIcon(icon_path))

        self.dataset_dir = os.path.join(script_dir, "dataset")
        self.extracted_dataset_dir = os.path.join(script_dir, "extracted_dataset")
        self.plot_dir = os.path.join(script_dir, "plot")

        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.extracted_dataset_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        container = QWidget()
        self.setCentralWidget(container)

        self.load_button = QPushButton("📁", container)
        self.load_button.clicked.connect(self.load_file)
        self.load_button.setFixedSize(30, 30)
        self.load_button.move(20, 10)

        self.save_button = QPushButton("💾", container)
        self.save_button.clicked.connect(self.save_forecast)
        self.save_button.setFixedSize(30, 30)
        self.save_button.move(60, 10)

        self.toggleSidePanelButton = QPushButton("⚙️", container)
        self.toggleSidePanelButton.clicked.connect(self.toggleSidePanel)
        self.toggleSidePanelButton.setFixedSize(30, 30)
        self.toggleSidePanelButton.move(100, 10)

        self.clear_button = QPushButton("🗑️", container)
        self.clear_button.clicked.connect(self.clear_all)
        self.clear_button.setFixedSize(30, 30)
        self.clear_button.move(140, 10)

        self.group_button = QPushButton("Group", container)
        self.group_button.clicked.connect(self.group_countries)
        self.group_button.setFixedSize(100, 30)
        self.group_button.move(500, 220)

        self.country_search = QLineEdit(container)
        self.country_search.setPlaceholderText("Search country...")
        self.country_search.setFixedSize(300, 30)
        self.country_search.move(20, 60)
        self.country_search.textChanged.connect(self.filter_country_list)

        self.country_list = QListWidget(container)
        self.country_list.setFixedSize(300, 100)
        self.country_list.move(20, 100)

        self.variable_combo = QComboBox(container)
        self.variable_combo.setFixedSize(200, 30)
        self.variable_combo.move(340, 60)

        self.forecasted_country_search = QLineEdit(container)
        self.forecasted_country_search.setPlaceholderText("Search forecast country...")
        self.forecasted_country_search.setFixedSize(350, 30)
        self.forecasted_country_search.move(560, 60)
        self.forecasted_country_search.textChanged.connect(self.filter_forecasted_country_list)

        self.forecasted_country_list = QListWidget(container)
        self.forecasted_country_list.setFixedSize(350, 100)
        self.forecasted_country_list.move(560, 100)

        self.start_year_label = QLabel("Start Year:", container)
        self.start_year_label.move(20, 220)
        self.start_year_spin = QSpinBox(container)
        self.start_year_spin.setRange(1900, 2100)
        self.start_year_spin.setFixedSize(80, 25)
        self.start_year_spin.move(100, 220)

        self.end_year_label = QLabel("End Year:", container)
        self.end_year_label.move(200, 220)
        self.end_year_spin = QSpinBox(container)
        self.end_year_spin.setRange(1900, 2100)
        self.end_year_spin.setFixedSize(80, 25)
        self.end_year_spin.move(280, 220)

        self.test_button = QPushButton("ADF Test", container)
        self.test_button.clicked.connect(self.run_adf_test)
        self.test_button.setFixedSize(100, 30)
        self.test_button.move(380, 220)

        self.plot_type_label = QLabel("Plot Type:", container)
        self.plot_type_label.move(20, 260)
        self.plot_combo = QComboBox(container)
        self.plot_combo.addItems(["Historical", "Forecast", "Both"])
        self.plot_combo.setFixedSize(150, 30)
        self.plot_combo.move(100, 260)

        self.chart_type_label = QLabel("Chart Type:", container)
        self.chart_type_label.move(260, 260)
        self.chart_type_combo = QComboBox(container)
        self.chart_type_combo.addItems(["Lines", "Stacked Bars"])
        self.chart_type_combo.setFixedSize(150, 30)
        self.chart_type_combo.move(340, 260)

        self.plot_button = QPushButton("Plot", container)
        self.plot_button.clicked.connect(self.plot_selected_data)
        self.plot_button.setFixedSize(100, 30)
        self.plot_button.move(500, 260)

        self.console = QTextEdit(container)
        self.console.setReadOnly(True)
        self.console.setFixedSize(900, 150)
        self.console.move(20, 300)

        self.canvas = FigureCanvas(Figure())
        self.canvas.setParent(container)
        self.canvas.setFixedSize(900, 400)
        self.canvas.move(20, 460)

        self.download_button = QPushButton("💾", container)
        self.download_button.setToolTip("Download plot as image")
        self.download_button.clicked.connect(self.download_plot)
        self.download_button.setFixedSize(30, 30)
        self.download_button.move(880, 470)

        self.df = None
        self.adf_results = None
        self.filtered_data = None
        self.sarimax_results = None
        self.arimax_results = None
        self.arima_results = None
        self.forecast_results = {}
        self.sidePanelWindow = None
        self.forecast_until_year = 2100
        self.replace_negative_forecast = False

    def closeEvent(self, event):
        if self.sidePanelWindow and self.sidePanelWindow.isVisible():
            self.sidePanelWindow.close()
        event.accept()

    def load_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load CSV File", self.dataset_dir, "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            try:
                self.df = pd.read_csv(file_name)
                self.update_combos()
                self.console.append(f"File {file_name} loaded successfully.")
            except Exception as e:
                self.console.append(f"Error loading file: {e}")

    def update_combos(self):
        if self.df is not None:
            countries = self.df['country'].unique().tolist()
            self.country_list.clear()
            self.forecasted_country_list.clear()

            for country in countries:
                country_item = QListWidgetItem(country)
                country_item.setFlags(country_item.flags() | Qt.ItemIsUserCheckable)
                country_item.setCheckState(Qt.Unchecked)
                self.country_list.addItem(country_item)

            variables = self.df.columns[2:].tolist()
            self.variable_combo.clear()
            self.variable_combo.addItems(variables)

            years = self.df['year'].unique()
            self.start_year_spin.setRange(int(years.min()), int(years.max()))
            self.end_year_spin.setRange(int(years.min()), int(years.max()))
            self.start_year_spin.setValue(int(years.min()))
            self.end_year_spin.setValue(int(years.max()))

    def toggleSidePanel(self):
        if self.sidePanelWindow and self.sidePanelWindow.isVisible():
            self.sidePanelWindow.hide()
        else:
            if not self.sidePanelWindow:
                self.sidePanelWindow = SidePanelWindow(self)
            main_window_geometry = self.frameGeometry()
            side_panel_x = main_window_geometry.x() + main_window_geometry.width()
            side_panel_y = main_window_geometry.y()

            side_panel_height = self.geometry().height()

            self.sidePanelWindow.move(side_panel_x, side_panel_y)
            self.sidePanelWindow.resize(self.sidePanelWindow.width(), side_panel_height)
            self.sidePanelWindow.show()

    def group_countries(self):
        selected_countries = self.get_selected_countries(self.country_list)
        if not selected_countries:
            self.console.append("Please select at least one country to group.")
            return
        self.group_panel = GroupPanelWindow(self)
        self.group_panel.show()

    def create_group(self, group_name):
        selected_countries = self.get_selected_countries(self.country_list)
        if not selected_countries:
            self.console.append("Please select at least one country to group.")
            return

        start_year = self.start_year_spin.value()
        end_year = self.end_year_spin.value()

        group_data = pd.DataFrame()
        for country in selected_countries:
            country_data = self.df[(self.df['country'] == country) & (self.df['year'] >= start_year) & (self.df['year'] <= end_year)]
            if group_data.empty:
                group_data = country_data.copy()
            else:
                group_data = group_data.set_index('year').add(country_data.set_index('year'), fill_value=0).reset_index()

        group_data['country'] = group_name

        self.df = pd.concat([self.df, group_data], ignore_index=True)

        self.update_combos()
        self.console.append(f"Group '{group_name}' created and added to the dataset.")
        self.country_search.setPlaceholderText("Search country...")

    def run_adf_test(self):
        if self.df is None:
            self.console.append("You must first load a CSV file.")
            return

        selected_countries = self.get_selected_countries(self.country_list)
        variable = self.variable_combo.currentText()
        start_year = self.start_year_spin.value()
        end_year = self.end_year_spin.value()

        if not selected_countries:
            self.console.append("Please select at least one country.")
            return

        self.adf_results = pd.DataFrame(columns=['Country', 'Variable', 'ADF Statistic', 'p-value', 'Num Lags', 'Num Observations', '1%', '5%', '10%', 'Differencing', 'p-value diff', 'Error'])

        for country in selected_countries:
            self.filtered_data = self.df[(self.df['country'] == country) & (self.df['year'] >= start_year) & (self.df['year'] <= end_year)].copy()

            if self.filtered_data.empty:
                self.console.append(f"No data for country {country} and selected year range.")
                continue

            self.filtered_data.set_index('year', inplace=True)

            adf_result = perform_adf_test_with_differencing(self.filtered_data[[variable]].dropna())
            if not adf_result.empty:
                adf_result['Country'] = country
                adf_result['Variable'] = variable
                self.adf_results = pd.concat([self.adf_results, adf_result], ignore_index=True)

        self.console.clear()
        self.console.append(self.format_adf_results(self.adf_results))

    def run_sarimax(self, p_range=None, d_range=None, q_range=None, seasonal_period=None):
        if self.adf_results is None or self.df is None:
            self.console.append("You must first run the ADF test.")
            return

        p_range = p_range if p_range is not None else range(0, 2)
        d_range = d_range if d_range is not None else range(0, 1)
        q_range = q_range if q_range is not None else range(0, 2)
        seasonal_period = seasonal_period if seasonal_period is not None else 11

        selected_countries = self.get_selected_countries(self.country_list)
        start_year = self.start_year_spin.value()
        end_year = self.end_year_spin.value()

        sarimax_results = optimize_sarimax_models(self.adf_results, self.df, selected_countries, p_range, d_range, q_range, seasonal_period, start_year, end_year)
        self.console.clear()
        self.console.append(self.format_sarimax_results(sarimax_results))

        forecast_results = forecast_future_sarimax(sarimax_results, self.df, start_year, self.forecast_until_year, self.replace_negative_forecast)
        self.forecast_results.update(forecast_results)

        self.apply_forecast_corrections()
        self.update_forecasted_countries_list()

    def run_arimax(self, p_range=None, d_range=None, q_range=None):
        if self.adf_results is None or self.df is None:
            self.console.append("You must first run the ADF test.")
            return

        p_range = p_range if p_range is not None else range(0, 2)
        d_range = d_range if d_range is not None else range(0, 1)
        q_range = q_range if q_range is not None else range(0, 2)

        selected_countries = self.get_selected_countries(self.country_list)
        start_year = self.start_year_spin.value()
        end_year = self.end_year_spin.value()

        for country in selected_countries:
            if self.adf_results[self.adf_results['Country'] == country].empty:
                self.console.append(f"No ADF results found for country {country}.")
                return

        arimax_results = optimize_arimax_models(self.adf_results, self.df, selected_countries, p_range, d_range, q_range, start_year, end_year)
        self.console.clear()
        self.console.append(self.format_arimax_results(arimax_results))

        forecast_results = forecast_future_arimax(arimax_results, self.df, start_year, self.forecast_until_year, self.replace_negative_forecast)
        self.forecast_results.update(forecast_results)

        self.apply_forecast_corrections()
        self.update_forecasted_countries_list()

    def run_arima(self, p_range=None, d_range=None, q_range=None):
        if self.adf_results is None or self.df is None:
            self.console.append("You must first run the ADF test.")
            return

        p_range = p_range if p_range is not None else range(0, 2)
        d_range = d_range if d_range is not None else range(0, 1)
        q_range = q_range if q_range is not None else range(0, 2)

        selected_countries = self.get_selected_countries(self.country_list)
        start_year = self.start_year_spin.value()
        end_year = self.end_year_spin.value()

        for country in selected_countries:
            if self.adf_results[self.adf_results['Country'] == country].empty:
                self.console.append(f"No ADF results found for country {country}.")
                return

        arima_results = optimize_arima_models(self.adf_results, self.df, selected_countries, p_range, d_range, q_range, start_year, end_year)
        self.console.clear()
        self.console.append(self.format_arima_results(arima_results))

        forecast_results = forecast_future_arima(arima_results, self.df, start_year, self.forecast_until_year, self.replace_negative_forecast)
        self.forecast_results.update(forecast_results)

        self.apply_forecast_corrections()
        self.update_forecasted_countries_list()

    def apply_forecast_corrections(self):
        if self.sidePanelWindow is not None:
            selected_country = self.get_selected_countries(self.country_list)
            if not selected_country:
                self.console.append("Please select a country in the country search list.")
                return

            target_year_text = self.sidePanelWindow.target_year_input.text()
            target_value_text = self.sidePanelWindow.target_value_input.text()
            continuous_correction = self.sidePanelWindow.continuous_correction_checkbox.isChecked()
            short_correction = self.sidePanelWindow.short_correction_checkbox.isChecked()
            start_correction = self.sidePanelWindow.start_correction_checkbox.isChecked()

            if target_year_text and target_value_text:
                target_year = int(target_year_text)
                target_value = float(target_value_text)
                variable = self.variable_combo.currentText()
                self.apply_linear_correction(selected_country[0], target_year, target_value, variable, continuous_correction, short_correction, start_correction)

    def apply_linear_correction(self, country, target_year, target_value, variable, continuous, short, start):
        forecast_key = None
        for key, value in self.forecast_results.items():
            if value['country'] == country:
                forecast_key = key
                break

        if not forecast_key:
            self.console.append(f"No forecast found for selected country: {country}")
            return

        forecast_data = self.forecast_results[forecast_key]
        forecast_values = forecast_data['forecast_values']
        forecast_years = forecast_values.index

        if target_year in forecast_years:
            if start:
                for year in forecast_years:
                    if year >= target_year:
                        forecast_values.loc[year] = target_value
            else:
                current_value = forecast_values.loc[target_year]
                correction_factor = (target_value - current_value) / (target_year - forecast_years.min())

                for year in forecast_years:
                    if year <= target_year:
                        forecast_values.loc[year] += correction_factor * (year - forecast_years.min())
                    else:
                        if continuous:
                            forecast_values.loc[year] += correction_factor * (year - forecast_years.min())
                        elif short:
                            forecast_values.loc[year] = target_value

            forecast_values[forecast_values < 0] = 0

    def update_forecasted_countries_list(self):
        self.forecasted_country_list.clear()
        forecasted_countries = list(self.forecast_results.keys())

        for forecast_key in forecasted_countries:
            forecasted_country_item = QListWidgetItem(forecast_key)
            forecasted_country_item.setFlags(forecasted_country_item.flags() | Qt.ItemIsUserCheckable)
            forecasted_country_item.setCheckState(Qt.Unchecked)
            self.forecasted_country_list.addItem(forecasted_country_item)

        self.forecasted_country_search.clear()

    def plot_selected_data(self):
        if self.forecast_results is None:
            self.console.append("You must first apply a model.")
            return

        plot_type = self.plot_combo.currentText()
        chart_type = self.chart_type_combo.currentText()
        selected_forecasts = self.get_selected_countries(self.forecasted_country_list)
        variable = self.variable_combo.currentText()

        if not selected_forecasts:
            self.console.append("Please select at least one forecast to plot.")
            return

        show_confidence_interval = self.sidePanelWindow.show_confidence_interval_checkbox.isChecked()
        if show_confidence_interval and len(selected_forecasts) > 1:
            self.console.append("The 'Show Confidence Interval' option is only available for one country at a time.")
            return

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        add_forecast_start_line = True
        if chart_type == "Lines":
            max_value = -float('inf')
            for forecast_key in selected_forecasts:
                max_value = max(max_value, plot_data(self.df, self.forecast_results, forecast_key, variable, plot_type, ax, add_forecast_start_line, show_confidence_interval))
                add_forecast_start_line = False
        elif chart_type == "Stacked Bars":
            max_value = plot_data_stacked_bar(self.df, self.forecast_results, selected_forecasts, variable, plot_type, ax)

        start_year = self.start_year_spin.value()
        end_year = self.forecast_until_year

        ax.set_xlim([start_year, end_year])
        ax.set_ylim([0, max_value * 1.01])

        self.canvas.draw()

    def apply_plot_settings(self, x_range, y_range, title, legend_size, xlabel, ylabel, xlabel_size, ylabel_size):
        if self.canvas.figure.axes:
            ax = self.canvas.figure.axes[0]
            if x_range and len(x_range) == 2:
                ax.set_xlim(x_range)
            if y_range and len(y_range) == 2:
                ax.set_ylim(y_range)
            if title:
                ax.set_title(title, fontsize=16, fontweight='bold')
            if legend_size:
                ax.legend(loc='upper left', fontsize=legend_size)
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=xlabel_size if xlabel_size else 14)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=ylabel_size if ylabel_size else 14)
            if xlabel_size and not xlabel:
                ax.xaxis.label.set_size(xlabel_size)
            if ylabel_size and not ylabel:
                ax.yaxis.label.set_size(ylabel_size)
            self.canvas.draw()

    def save_forecast(self):
        selected_countries = self.get_selected_countries(self.forecasted_country_list)
        if not selected_countries:
            self.console.append("Please select at least one forecasted country to save.")
            return

        variable = self.variable_combo.currentText()
        forecast_data = pd.DataFrame()
        for country in selected_countries:
            historical_data = self.df[(self.df['country'] == country)][['country', 'year', variable]]
            forecast_values = self.forecast_results[country]['forecast_values']
            forecast_years = forecast_values.index
            forecast_df = pd.DataFrame({
                'country': country,
                'year': forecast_years,
                variable: forecast_values.values
            })
            combined_data = pd.concat([historical_data, forecast_df], ignore_index=True)
            combined_data = combined_data.drop_duplicates(subset=['year'], keep='last')
            forecast_data = pd.concat([forecast_data, combined_data], ignore_index=True)

        save_path, _ = QFileDialog.getSaveFileName(self, "Save CSV File", self.extracted_dataset_dir, "CSV Files (*.csv);;All Files (*)")
        if save_path:
            forecast_data.to_csv(save_path, index=False)
            self.console.append(f"Forecast saved to {save_path}")

    def download_plot(self):
        options = QFileDialog.Options()
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Plot Image", self.plot_dir, "PNG Files (*.png);;All Files (*)", options=options)
        if save_path:
            self.canvas.figure.savefig(save_path)
            self.canvas.figure.savefig(save_path, dpi=400)
            self.console.append(f"Plot saved to {save_path}")

    def filter_country_list(self):
        filter_text = self.country_search.text().lower()
        for index in range(self.country_list.count()):
            item = self.country_list.item(index)
            item.setHidden(filter_text not in item.text().lower())

    def filter_forecasted_country_list(self):
        filter_text = self.forecasted_country_search.text().lower()
        for index in range(self.forecasted_country_list.count()):
            item = self.forecasted_country_list.item(index)
            item.setHidden(filter_text not in item.text().lower())

    def format_adf_results(self, adf_results):
        if adf_results.empty:
            return "No results to display."
        return adf_results.to_html(index=False)

    def format_sarimax_results(self, sarimax_results):
        formatted_results = ""
        for country, result in sarimax_results.items():
            if 'model_object' in result:
                summary_html = result['model_summary'].as_html()
                formatted_results += f"<b>SARIMAX results for {country}:</b><br>{summary_html}<br>"
            else:
                formatted_results += f"<b>Failed to model {country}:</b> {result['error']}<br>"
        return formatted_results

    def format_arimax_results(self, arimax_results):
        formatted_results = ""
        for country, result in arimax_results.items():
            if 'model_object' in result:
                summary_html = result['model_summary'].as_html()
                formatted_results += f"<b>ARIMAX results for {country}:</b><br>{summary_html}<br>"
            else:
                formatted_results += f"<b>Failed to model {country}:</b> {result['error']}<br>"
        return formatted_results

    def format_arima_results(self, arima_results):
        formatted_results = ""
        for country, result in arima_results.items():
            if 'model_object' in result:
                summary_html = result['model_summary'].as_html()
                formatted_results += f"<b>ARIMA results for {country}:</b><br>{summary_html}<br>"
            else:
                formatted_results += f"<b>Failed to model {country}:</b> {result['error']}<br>"
        return formatted_results

    def get_selected_countries(self, list_widget):
        selected_countries = []
        for index in range(list_widget.count()):
            item = list_widget.item(index)
            if item.checkState() == Qt.Checked:
                selected_countries.append(item.text())
        return selected_countries

    def clear_all(self):
        selected_forecasts = self.get_selected_countries(self.forecasted_country_list)
        if not selected_forecasts:
            self.console.append("No forecasted countries selected to delete.")
            return

        for forecast_key in selected_forecasts:
            if forecast_key in self.forecast_results:
                del self.forecast_results[forecast_key]

        self.update_forecasted_countries_list()
        self.console.append("Selected models successfully deleted.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
