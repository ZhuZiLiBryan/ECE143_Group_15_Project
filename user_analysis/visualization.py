"""Visualization helpers for the user model pipeline."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIG_SIZE, PLOT_STYLE
from .data_loader import load_transactions
from .features import engineer_features
from .models import train_and_evaluate

sns.set_style(PLOT_STYLE)


def plot_feature_importance(feature_importance):
    """
    Render the horizontal bar chart for feature importance ranking.
    
    Args:
        feature_importance: DataFrame with 'feature' and 'importance' columns
        
    Returns:
        None. Displays the plot using plt.show()
    """
    plt.figure(figsize=FIG_SIZE)
    plt.barh(
        range(len(feature_importance)),
        feature_importance["importance"],
        align="center",
    )
    plt.yticks(range(len(feature_importance)), feature_importance["feature"])
    plt.xlabel("importance", fontsize=12)
    plt.ylabel("feature", fontsize=12)
    plt.title("Random Forest - Feature Importance Ranking", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_feature_importance(data_path: str | None = None):
    """
    Convenience wrapper for reproducing the notebook's feature-importance visualization.
    
    Loads data, performs feature engineering, trains the random forest, and
    displays the bar chart so external callers can obtain the plot with a single call.
    
    Args:
        data_path: Optional path to CSV file. If None, uses default path
        
    Returns:
        None. Displays the feature importance bar chart
    """
    df = load_transactions(data_path)
    _, X, y, _, _ = engineer_features(df)
    results = train_and_evaluate(X, y)
    feature_importance = results["feature_importance"]
    plot_feature_importance(feature_importance)

