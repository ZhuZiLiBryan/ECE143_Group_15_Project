"""Model training sub-package for user model pipeline."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import RANDOM_STATE, TEST_SIZE
from .evaluation import ModelResult, compute_balanced_class_weights, summarize_result
from .decision_tree import train_decision_tree
from .random_forest import train_random_forest


def train_and_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train Decision Tree and Random Forest models and capture metrics.
    
    Args:
        X: Feature matrix DataFrame
        y: Target labels Series
        
    Returns:
        Dictionary containing:
            - splits: Train/test split data
            - class_weights: Computed balanced class weights
            - decision_tree: ModelResult for Decision Tree
            - random_forest: ModelResult for Random Forest
            - rf_model: Fitted RandomForestClassifier
            - feature_importance: DataFrame with feature importance rankings
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    class_weights = compute_balanced_class_weights(y)

    _, dt_result = train_decision_tree(X_train, y_train, X_test, y_test)
    rf_model, rf_result = train_random_forest(X_train, y_train, X_test, y_test)

    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return {
        "splits": {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
        "class_weights": class_weights,
        "decision_tree": dt_result,
        "random_forest": rf_result,
        "rf_model": rf_model,
        "feature_importance": feature_importance,
    }


__all__ = [
    "train_and_evaluate",
    "train_decision_tree",
    "train_random_forest",
    "ModelResult",
    "compute_balanced_class_weights",
    "summarize_result",
]

