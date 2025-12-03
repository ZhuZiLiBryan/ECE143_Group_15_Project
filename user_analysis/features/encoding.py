"""Categorical encoding utilities."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_FEATURES: List[str] = ["customer_favorite_coffee", "last_coffee"]


def encode_categoricals(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df: DataFrame with customer_favorite_coffee, last_coffee columns
        
    Returns:
        Tuple containing:
            - DataFrame with added encoded columns (suffixed with _encoded)
            - Dictionary mapping column names to fitted LabelEncoder objects
    """
    df = df.copy()
    encoders: Dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders

