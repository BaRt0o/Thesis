import numpy as np
import pandas as pd

def forecast_future(sarimax_results, df):
    forecast_results = {}

    for country, result in sarimax_results.items():
        if 'model_object' in result:
            filtered_data = df[df['country'] == country]
            last_data_year = filtered_data['year'].max()
            
            if pd.isnull(last_data_year) or not isinstance(last_data_year, (int, np.integer)):
                raise ValueError(f"El último año de los datos filtrados es inválido: {last_data_year}")
            
            forecast_years = pd.date_range(start=pd.to_datetime(str(int(last_data_year) + 1)), end=pd.to_datetime('2101'), freq='A').year
            steps_to_2100 = len(forecast_years)
            model = result['model_object']
            forecast = model.get_forecast(steps=steps_to_2100)
            forecast_values = forecast.predicted_mean
            forecast_ci = forecast.conf_int(alpha=0.05)
            forecast_ci.columns = ['mean_ci_lower', 'mean_ci_upper']
            forecast_values.index = forecast_years
            forecast_ci.index = forecast_years
            forecast_results[country] = {
                'forecast_values': forecast_values,
                'forecast_ci': forecast_ci
            }
    
    return forecast_results
