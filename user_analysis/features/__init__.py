"""Feature engineering sub-package for user model pipeline."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .temporal import add_temporal_columns, add_cyclical_columns
from .customer import add_customer_history, add_last_purchase
from .price import add_price_context, add_interactions
from .encoding import encode_categoricals, CATEGORICAL_FEATURES

# Base numeric features assembled prior to categorical encodings.
NUMERIC_BASE_FEATURES: List[str] = [
    "hour",
    "day_of_week_num",
    "month_num",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "customer_visit_count",
    "customer_avg_spend",
    "avg_price_by_hour",
    "avg_price_by_month",
    "hour_weekend",
    "month_weekend",
]


def engineer_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str], Dict[str, LabelEncoder]]:
    """
    Run the complete feature engineering stack.
    
    Args:
        df: Raw transaction DataFrame with datetime, money, coffee_name columns
        
    Returns:
        Tuple containing:
            - engineered_df: Original dataframe with appended engineered columns
            - X: Feature matrix ready for modeling
            - y: Target labels (coffee_name)
            - feature_names: Names of numeric (including encoded categorical) features
            - encoders: Fitted label encoders for categorical columns
    """
    engineered = df.copy()
    engineered = add_temporal_columns(engineered)
    engineered = add_customer_history(engineered)
    engineered = add_last_purchase(engineered)
    engineered = add_cyclical_columns(engineered)
    engineered = add_price_context(engineered)
    engineered = add_interactions(engineered)

    encoded_df, encoders = encode_categoricals(engineered)
    feature_names = NUMERIC_BASE_FEATURES + [
        f"{col}_encoded" for col in CATEGORICAL_FEATURES
    ]
    X = encoded_df[feature_names].fillna(0)
    y = encoded_df["coffee_name"]
    return encoded_df, X, y, feature_names, encoders


__all__ = [
    "engineer_features",
    "NUMERIC_BASE_FEATURES",
    "CATEGORICAL_FEATURES",
    "add_temporal_columns",
    "add_cyclical_columns",
    "add_customer_history",
    "add_last_purchase",
    "add_price_context",
    "add_interactions",
    "encode_categoricals",
]

