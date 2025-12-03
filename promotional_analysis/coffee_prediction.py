"""
Predicting most popular coffee types using time series models.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


def predict_most_sold_coffee_month(df: pd.DataFrame, datetime_col: str = 'datetime',
                            coffee_col: str = 'coffee_name', 
                            months_back: int = 12,
                            order: tuple = (1, 1, 1),
                            seasonal_order: tuple = (1, 1, 1, 12)) -> tuple:
    """
    Predict which coffee will be most sold in the next month.
    
    Args:
        df: Input DataFrame with sales data
        datetime_col: Name of the datetime column
        coffee_col: Name of the coffee name column
        months_back: Number of months to use for training
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal ARIMA order (P, D, Q, s)
        
    Returns:
        Tuple of (most_sold_coffee_name, predicted_sales, all_predictions_dict)
    """
 
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    monthly_coffee_sales = df.groupby([pd.Grouper(key=datetime_col, freq='M'), 
                                       coffee_col]).size().unstack(fill_value=0)
    
    # Take the last N months as training data
    last_months_coffee = monthly_coffee_sales.tail(months_back)
    
    # Fit SARIMA for each coffee type and predict next month's sales
    predictions = {}
    for coffee in last_months_coffee.columns:
        series = last_months_coffee[coffee]
        
        # Handle all-zero columns or insufficient data
        if series.sum() == 0 or series.count() < 2:
            predictions[coffee] = 0
            continue
        
        try:
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
            pred = model_fit.forecast(steps=1).iloc[0]
            predictions[coffee] = max(pred, 0)  # Ensure non-negative
        except Exception:
            predictions[coffee] = 0  # Fallback for failed models
    
    # Find the coffee with the highest predicted sales
    most_sold_coffee = max(predictions, key=predictions.get)
    
    return most_sold_coffee, predictions[most_sold_coffee], predictions


def predict_most_sold_coffee_week(df: pd.DataFrame, datetime_col: str = 'datetime',
                            coffee_col: str = 'coffee_name', 
                            weeks_back: int = 4,
                            order: tuple = (1, 1, 1),
                            seasonal_order: tuple = (1, 1, 1, 12)) -> tuple:
    """
    Predict which coffee will be most sold in the next week.
    """
    df = df.copy()  
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    weekly_coffee_sales = df.groupby([pd.Grouper(key=datetime_col, freq='W'), 
                                          coffee_col]).size().unstack(fill_value=0)
    
    # Take the last N weeks as training data
    last_weeks_coffee = weekly_coffee_sales.tail(weeks_back)
    
    # Fit SARIMA for each coffee type and predict next week's sales
    predictions = {}
    for coffee in last_weeks_coffee.columns:
        series = last_weeks_coffee[coffee]
        if series.sum() == 0 or series.count() < 2:
            predictions[coffee] = 0
            continue
        try:
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
            model_fit = model.fit(disp=False)
            pred = model_fit.forecast(steps=1).iloc[0]
            predictions[coffee] = max(pred, 0)
        except Exception:
            predictions[coffee] = 0
    
    # Find the coffee with the highest predicted sales
    most_sold_coffee = max(predictions, key=predictions.get)
    
    return most_sold_coffee, predictions[most_sold_coffee], predictions