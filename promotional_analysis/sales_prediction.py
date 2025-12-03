"""
Sales forecasting using SARIMA models.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


def predict_next_day_sales(daily_sales: pd.Series, training_days: int = 365,
                           order: tuple = (1, 1, 1), 
                           seasonal_order: tuple = (1, 1, 1, 12)) -> float:
    """
    Predict sales for the next day using SARIMA model.
    
    Args:
        daily_sales: Series with daily sales data
        training_days: Number of days to use for training
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal ARIMA order (P, D, Q, s)
        
    Returns:
        Predicted sales value for next day
    """
    # Use the last N days as training data
    training_data = daily_sales.last(f'{training_days}D')
    
    # Fit SARIMA model
    model = SARIMAX(training_data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    
    # Forecast next day
    forecast = model_fit.forecast(steps=1)
    return forecast.iloc[0]


def calculate_moving_average(series: pd.Series, window: int = 7) -> pd.Series:
    """
    Calculate moving average of a time series.
    
    Args:
        series: Input time series
        window: Window size for moving average
        
    Returns:
        Series with moving average values
    """
    return series.rolling(window=window, min_periods=1).mean()

