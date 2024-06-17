import pandas as pd
from statsmodels.tsa.stattools import adfuller

def perform_adf_test_with_differencing(data, max_differencing=2):
    """
    Perform ADF test with differencing up to a maximum number of differencing steps.
    
    Args:
        data (pd.DataFrame): Data for the ADF test.
        max_differencing (int): Maximum number of differencing steps.

    Returns:
        pd.DataFrame: ADF test results.
    """
    if data.nunique().values[0] == 1:
        return pd.DataFrame([{
            'ADF Statistic': None,
            'p-value': None,
            'N° Lags': None,
            'N° Observations': None,
            '1%': None,
            '5%': None,
            '10%': None,
            'Differencing': 0,
            'p-value diff': None,
            'Error': 'Input data is constant'
        }])

    original_adf_result = adfuller(data)
    result = {
        'ADF Statistic': original_adf_result[0],
        'p-value': original_adf_result[1],
        'N° Lags': original_adf_result[2],
        'N° Observations': original_adf_result[3],
        '1%': original_adf_result[4]['1%'],
        '5%': original_adf_result[4]['5%'],
        '10%': original_adf_result[4]['10%'],
        'Differencing': 0,
        'p-value diff': None
    }

    for i in range(1, max_differencing + 1):
        data = data.diff().dropna()
        if data.nunique().values[0] == 1:
            result.update({
                'ADF Statistic': None,
                'p-value diff': None,
                'N° Lags': None,
                'N° Observations': None,
                '1%': None,
                '5%': None,
                '10%': None,
                'Differencing': i,
                'Error': 'Input data is constant after differencing'
            })
            break

        adf_test = adfuller(data)
        if adf_test[1] <= 0.05 or i == max_differencing:
            result.update({
                'ADF Statistic': adf_test[0],
                'p-value diff': adf_test[1],
                'N° Lags': adf_test[2],
                'N° Observations': adf_test[3],
                '1%': adf_test[4]['1%'],
                '5%': adf_test[4]['5%'],
                '10%': adf_test[4]['10%'],
                'Differencing': i
            })
            break

    return pd.DataFrame([result])
