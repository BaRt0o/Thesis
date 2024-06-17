import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_data(df, forecast_results, forecast_key, variable, plot_type, ax, add_forecast_start_line, show_confidence_interval):
    """
    Plot data and forecasts on a matplotlib axis.

    Args:
        df (pd.DataFrame): Data frame containing the data.
        forecast_results (dict): Forecast results.
        forecast_key (str): Key for the forecast result.
        variable (str): Variable to plot.
        plot_type (str): Type of plot ("Historical", "Forecast", "Both").
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on.
        add_forecast_start_line (bool): Whether to add a line indicating the start of the forecast.
        show_confidence_interval (bool): Whether to show confidence intervals.

    Returns:
        float: Maximum value in the plotted data.
    """
    forecast = forecast_results[forecast_key]
    country = forecast['country']
    historical_data = df[df['country'] == country][['year', variable]].set_index('year')

    min_year = int(historical_data.index.min())
    max_year = int(historical_data.index.max())

    combined_data = pd.DataFrame(index=range(min_year, 2101))
    combined_data[variable] = historical_data[variable]

    if forecast_key in forecast_results:
        forecast_values = forecast_results[forecast_key]['forecast_values']
        forecast_ci = forecast_results[forecast_key]['forecast_ci']
        forecast_index = forecast_values.index

        combined_data.loc[forecast_index, variable] = forecast_values

    historical_part = combined_data.loc[:max_year, variable].dropna()
    forecast_part = combined_data.loc[max_year+1:, variable].dropna()

    max_value = max(historical_part.max(), forecast_part.max())

    if plot_type == "Historical":
        ax.plot(historical_part.index, historical_part, label=f'Historical - {country}')
    elif plot_type == "Forecast":
        ax.plot(forecast_part.index, forecast_part, label=f'Forecast - {country}', linestyle='--')
        if show_confidence_interval:
            ax.fill_between(forecast_ci.index, forecast_ci['mean_ci_lower'], forecast_ci['mean_ci_upper'], color='gray', alpha=0.2)
    elif plot_type == "Both":
        combined_part = pd.concat([historical_part, forecast_part])
        ax.plot(combined_part.index, combined_part, label=country)
        if show_confidence_interval:
            ax.fill_between(forecast_ci.index, forecast_ci['mean_ci_lower'], forecast_ci['mean_ci_upper'], color='gray', alpha=0.2)

    if add_forecast_start_line:
        ax.axvline(x=max_year + 0.5, color='black', linestyle='--', linewidth=1, label='Forecast Start')
        ax.axvline(x=max_year + 30.5, color='red', linestyle='--', linewidth=1, label='Cop28 Agreement')
    ax.set_title(f'{variable} Production ({plot_type})', fontsize=16, fontweight='bold')
    ax.set_ylabel('Production (TWh)', fontsize=14)
    ax.set_xlabel('Year', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', which='both', color='grey', alpha=0.5)
    ax.set_ylim(bottom=0)

    return max_value

def plot_data_stacked_bar(df, forecast_results, forecast_keys, variable, plot_type, ax):
    """
    Plot stacked bar chart for data and forecasts on a matplotlib axis.

    Args:
        df (pd.DataFrame): Data frame containing the data.
        forecast_results (dict): Forecast results.
        forecast_keys (list): List of forecast keys.
        variable (str): Variable to plot.
        plot_type (str): Type of plot ("Historical", "Forecast", "Both").
        ax (matplotlib.axes.Axes): Matplotlib axis to plot on.

    Returns:
        float: Maximum value in the plotted data.
    """
    combined_data = pd.DataFrame()

    for forecast_key in forecast_keys:
        forecast = forecast_results[forecast_key]
        country = forecast['country']
        historical_data = df[df['country'] == country][['year', variable]].set_index('year')
        temp_combined_data = pd.DataFrame(index=range(int(historical_data.index.min()), 2101))
        temp_combined_data[variable] = historical_data[variable]

        if forecast_key in forecast_results:
            forecast_values = forecast_results[forecast_key]['forecast_values']
            forecast_index = forecast_values.index
            temp_combined_data.loc[forecast_index, variable] = forecast_values

        combined_data[country] = temp_combined_data[variable]

    combined_data = combined_data.fillna(0)

    ax.clear()

    bottom = pd.Series([0] * len(combined_data.index), index=combined_data.index)
    colors = plt.cm.tab20(np.linspace(0, 1, len(forecast_keys)))
    max_value = combined_data.sum(axis=1).max()

    for i, forecast_key in enumerate(forecast_keys):
        forecast = forecast_results[forecast_key]
        country = forecast['country']
        ax.bar(combined_data.index, combined_data[country], bottom=bottom, color=colors[i], label=country)
        bottom += combined_data[country]

    ax.axvline(x=2022.5, color='black', linestyle='--', linewidth=1, label='Forecast Start')
    ax.axvline(x=2050.5, color='red', linestyle='--', linewidth=1, label='Cop28 Agreement')
    ax.set_title(f'{variable} Production ({plot_type})', fontsize=16, fontweight='bold')
    ax.set_ylabel('Production (TWh)', fontsize=14)
    ax.set_xlabel('Year', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', which='both', color='grey', alpha=0.5)
    ax.set_ylim(bottom=0)

    return max_value
