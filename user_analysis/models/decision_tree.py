"""Decision Tree model training."""
from __future__ import annotations

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from ..config import RANDOM_STATE
from .evaluation import ModelResult, summarize_result


def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    max_depth: int = 5,
) -> tuple[DecisionTreeClassifier, ModelResult]:
    """
    Train and evaluate a Decision Tree classifier.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        X_test: Test feature matrix
        y_test: Test labels
        max_depth: Maximum tree depth
        
    Returns:
        Tuple containing fitted model and ModelResult
    """
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = summarize_result("Decision Tree", y_test, y_pred)
    return model, result

