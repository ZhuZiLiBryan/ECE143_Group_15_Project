"""Random Forest model training."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ..config import RANDOM_STATE
from .evaluation import ModelResult, summarize_result


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 15,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
) -> tuple[RandomForestClassifier, ModelResult]:
    """
    Train and evaluate a Random Forest classifier.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        X_test: Test feature matrix
        y_test: Test labels
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples in leaf
        
    Returns:
        Tuple containing fitted model and ModelResult
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = summarize_result("Random Forest", y_test, y_pred)
    return model, result

