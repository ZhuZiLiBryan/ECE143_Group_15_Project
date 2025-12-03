"""Data loading utilities for the user model pipeline."""
from __future__ import annotations

import pandas as pd

from .config import resolve_data_path


def load_transactions(data_path: str | None = None) -> pd.DataFrame:
    """
    Load the raw transaction data and parse datetime columns.
    
    Args:
        data_path: Optional path to the CSV file. If None, uses default path
        
    Returns:
        DataFrame containing transaction data with parsed datetime columns
    """
    path = resolve_data_path(data_path)
    df = pd.read_csv(path)

    # Normalize datetime columns for downstream processing.
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df

