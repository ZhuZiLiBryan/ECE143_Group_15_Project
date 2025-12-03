"""Model evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight


@dataclass
class ModelResult:
    """Container for model evaluation results."""
    name: str
    accuracy: float
    report: str


def compute_balanced_class_weights(y: pd.Series) -> Dict[str, float]:
    """
    Compute balanced class weights for handling imbalanced classes.
    
    Args:
        y: Target labels Series
        
    Returns:
        Dictionary mapping class names to their computed weights
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def summarize_result(name: str, y_true: pd.Series, y_pred: np.ndarray) -> ModelResult:
    """
    Summarize model evaluation results.
    
    Args:
        name: Model name string
        y_true: True target labels
        y_pred: Predicted labels array
        
    Returns:
        ModelResult dataclass with name, accuracy, and classification report
    """
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    return ModelResult(name=name, accuracy=accuracy, report=report)

