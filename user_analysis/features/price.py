"""Price and interaction feature engineering functions."""
from __future__ import annotations

import pandas as pd


def add_price_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based price features using historical averages.
    
    Args:
        df: DataFrame with hour, month_num, money columns
        
    Returns:
        DataFrame with added avg_price_by_hour, avg_price_by_month columns
    """
    df = df.copy()
    if "money" not in df.columns:
        raise KeyError("Expected 'money' column in dataframe.")

    df["avg_price_by_hour"] = df.groupby("hour")["money"].transform("mean")
    df["avg_price_by_month"] = df.groupby("month_num")["money"].transform("mean")
    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features between time and weekend.
    
    Args:
        df: DataFrame with hour, month_num, is_weekend columns
        
    Returns:
        DataFrame with added hour_weekend, month_weekend columns
    """
    df = df.copy()
    df["hour_weekend"] = df["hour"] * df["is_weekend"]
    df["month_weekend"] = df["month_num"] * df["is_weekend"]
    return df

