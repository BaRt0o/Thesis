import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def optimize_sarimax(series, p_range, d_range, q_range, seasonal_period):
    """
    Optimize SARIMAX model parameters.

    Args:
        series (pd.Series): Time series data.
        p_range (range): Range of p values.
        d_range (range): Range of d values.
        q_range (range): Range of q values.
        seasonal_period (int): Seasonal period.

    Returns:
        tuple: Best AIC, best order, best seasonal order, best model.
    """
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    best_mdl = None

    P = D = Q = range(2)
    m = seasonal_period

    for p in p_range:
        for d in d_range:
            for q in q_range:
                for P_ in P:
                    for D_ in D:
                        for Q_ in Q:
                            try:
                                temp_model = SARIMAX(series,
                                                     order=(p, d, q),
                                                     seasonal_order=(P_, D_, Q_, m),
                                                     enforce_stationarity=False,
                                                     enforce_invertibility=False)
                                results = temp_model.fit(disp=False)
                                if results.aic < best_aic:
                                    best_aic = results.aic
                                    best_order = (p, d, q)
                                    best_seasonal_order = (P_, D_, Q_, m)
                                    best_mdl = results
                            except:
                                continue
    return best_aic, best_order, best_seasonal_order, best_mdl

def optimize_sarimax_models(adf_results, df, selected_countries, p_range, d_range, q_range, seasonal_period, start_year, end_year):
    """
    Optimize SARIMAX models for multiple countries.

    Args:
        adf_results (pd.DataFrame): ADF test results.
        df (pd.DataFrame): Data frame containing the data.
        selected_countries (list): List of selected countries.
        p_range (range): Range of p values.
        d_range (range): Range of d values.
        q_range (range): Range of q values.
        seasonal_period (int): Seasonal period.
        start_year (int): Start year for the data.
        end_year (int): End year for the data.

    Returns:
        dict: SARIMAX results for each country.
    """
    sarimax_results = {}

    for country in selected_countries:
        variable = adf_results[adf_results['Country'] == country]['Variable'].values[0]
        data_series = df[(df['country'] == country) & (df['year'] >= start_year) & (df['year'] <= end_year) & (df[variable].notna())][variable]

        if data_series.empty or len(data_series) < max(p_range) + max(d_range) + max(q_range) + 1:
            sarimax_results[country] = {'error': 'Insufficient data for modeling.'}
            continue

        try:
            aic, order, seasonal_order, model = optimize_sarimax(data_series, p_range, d_range, q_range, seasonal_period)
            if model is not None:
                sarimax_results[country] = {
                    'aic': aic, 
                    'order': order, 
                    'seasonal_order': seasonal_order, 
                    'model_summary': model.summary(),
                    'model_object': model
                }
            else:
                sarimax_results[country] = {'error': 'Model optimization failed.'}
        except Exception as e:
            sarimax_results[country] = {'error': str(e)}

    return sarimax_results

def forecast_future(sarimax_results, df, start_year, forecast_until_year=2100, replace_negative_forecast=False):
    """
    Forecast future values using SARIMAX models.

    Args:
        sarimax_results (dict): SARIMAX results.
        df (pd.DataFrame): Data frame containing the data.
        start_year (int): Start year for the forecast.
        forecast_until_year (int): Year until which to forecast.
        replace_negative_forecast (bool): Whether to replace negative forecast values with zero.

    Returns:
        dict: Forecast results for each country.
    """
    forecast_results = {}

    for country, result in sarimax_results.items():
        if 'model_object' in result:
            filtered_data = df[(df['country'] == country) & (df['year'] >= start_year)]
            last_data_year = filtered_data['year'].max()

            if pd.isnull(last_data_year) or not isinstance(last_data_year, (int, np.integer)):
                raise ValueError(f"The last year of the filtered data is invalid: {last_data_year}")

            forecast_years = pd.date_range(start=pd.to_datetime(str(int(last_data_year) + 1)), end=pd.to_datetime(str(forecast_until_year + 1)), freq='A').year
            steps_to_forecast_until = len(forecast_years)
            model = result['model_object']
            forecast = model.get_forecast(steps=steps_to_forecast_until)
            forecast_values = forecast.predicted_mean
            forecast_ci = forecast.conf_int(alpha=0.05)
            forecast_ci.columns = ['mean_ci_lower', 'mean_ci_upper']
            forecast_values.index = forecast_years
            forecast_ci.index = forecast_years

            if replace_negative_forecast:
                forecast_values[forecast_values < 0] = 0

            forecast_key = f"{country} ({forecast_until_year}) - SARIMAX {result['order']} ({result['seasonal_order'][3]})"
            forecast_results[forecast_key] = {
                'forecast_values': forecast_values,
                'forecast_ci': forecast_ci,
                'country': country,
                'model': 'SARX',
                'order': result['order'],
                'seasonal_order': result['seasonal_order'],
                'forecast_until_year': forecast_until_year
            }

    return forecast_results
