"""Temporal feature engineering functions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal feature columns derived from datetime.
    
    Args:
        df: DataFrame with 'datetime' column
        
    Returns:
        DataFrame with added hour, day_of_week_num, month_num, day_of_week_name, is_weekend columns
    """
    df = df.copy()
    if "datetime" not in df.columns:
        raise KeyError("Expected 'datetime' column in dataframe.")

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week_num"] = df["datetime"].dt.dayofweek
    df["month_num"] = df["datetime"].dt.month
    df["day_of_week_name"] = df["datetime"].dt.day_name()
    df["is_weekend"] = df["day_of_week_name"].isin(["Saturday", "Sunday"]).astype(int)
    return df


def add_cyclical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical time features using sin/cos encoding.
    
    Args:
        df: DataFrame with hour, month_num, day_of_week_num columns
        
    Returns:
        DataFrame with added hour_sin, hour_cos, month_sin, month_cos, day_of_week_sin, day_of_week_cos columns
    """
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week_num"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week_num"] / 7)
    return df

