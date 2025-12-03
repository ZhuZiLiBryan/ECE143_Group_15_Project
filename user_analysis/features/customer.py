"""Customer-related feature engineering functions."""
from __future__ import annotations

import pandas as pd


def add_customer_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add customer historical preference features for card customers.
    
    Args:
        df: DataFrame with cash_type, card, coffee_name, datetime, money columns
        
    Returns:
        DataFrame with added customer_favorite_coffee, customer_visit_count, customer_avg_spend columns
    """
    df = df.copy()
    if "cash_type" not in df.columns:
        df["cash_type"] = ""

    card_df = df[df["cash_type"] == "card"].dropna(subset=["card"]).copy()
    if not card_df.empty:
        customer_stats = card_df.groupby("card").agg(
            customer_favorite_coffee=("coffee_name", _first_mode),
            customer_visit_count=("datetime", "count"),
            customer_avg_spend=("money", "mean"),
        )
        df = df.merge(customer_stats, on="card", how="left")
        df["customer_favorite_coffee"] = df["customer_favorite_coffee"].fillna("Unknown")
        df["customer_visit_count"] = df["customer_visit_count"].fillna(0)
        df["customer_avg_spend"] = df["customer_avg_spend"].fillna(df["money"].mean())
    else:
        df["customer_favorite_coffee"] = "Unknown"
        df["customer_visit_count"] = 0
        df["customer_avg_spend"] = df["money"].mean()
    return df


def add_last_purchase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time series feature for previous coffee purchase.
    
    Args:
        df: DataFrame with card, datetime, coffee_name columns
        
    Returns:
        DataFrame sorted by card/datetime with added last_coffee column
    """
    df = df.copy()
    sort_cols = []
    if "card" in df.columns:
        sort_cols.append("card")
    sort_cols.append("datetime")
    df_sorted = df.sort_values(sort_cols)
    group_col = "card" if "card" in df.columns else None
    if group_col:
        df_sorted["last_coffee"] = df_sorted.groupby(group_col)["coffee_name"].shift(1)
    else:
        df_sorted["last_coffee"] = "Unknown"
    df_sorted["last_coffee"] = df_sorted["last_coffee"].fillna("Unknown")
    return df_sorted


def _first_mode(series: pd.Series) -> str:
    """
    Get the first mode value from a Series.
    
    Args:
        series: Pandas Series to compute mode from
        
    Returns:
        First mode value as string, or "Unknown" if series is empty
    """
    mode = series.mode()
    return mode.iat[0] if not mode.empty else "Unknown"

