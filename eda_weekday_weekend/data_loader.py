"""Data loading utilities for the weekday/weekend EDA package."""
from __future__ import annotations

import holidays
import pandas as pd

from .config import resolve_data_path


def load_and_preprocess(data_path: str | None = None) -> pd.DataFrame:
    """
    Load raw transaction data, parse datetime columns, and classify day types.
    
    Args:
        data_path: Optional path to the CSV file. If None, uses default path
        
    Returns:
        DataFrame with added weekday and day_type columns
    """
    path = resolve_data_path(data_path)
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["weekday"] = df["datetime"].dt.weekday  # 0=Monday, 6=Sunday

    us_holidays = holidays.US(years=[2024, 2025])

    def classify_day_type(row: pd.Series) -> str:
        date = row["date"]
        weekday = row["weekday"]
        if date.date() in us_holidays:
            return "Holiday"
        elif weekday >= 5:
            return "Weekend"
        else:
            return "Weekday"

    df["day_type"] = df.apply(classify_day_type, axis=1)
    return df


def compute_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily sales and order counts by date and day_type.
    
    Args:
        df: Preprocessed DataFrame with day_type column
        
    Returns:
        DataFrame with date, day_type, total_sales, order_count columns
    """
    daily = df.groupby(["date", "day_type"]).agg({"money": ["sum", "count"]}).reset_index()
    daily.columns = ["date", "day_type", "total_sales", "order_count"]
    return daily

