"""Executable script that mirrors the original model_user notebook."""
from __future__ import annotations

from pathlib import Path

from .data_loader import load_transactions
from .features import engineer_features
from .models import train_and_evaluate
from .visualization import plot_feature_importance


def main(data_path: str | Path | None = None, show_plot: bool = True) -> None:
    """
    Run the complete user model pipeline mirroring the original notebook.
    
    Args:
        data_path: Optional path to CSV file. If None, uses default path
        show_plot: Whether to display feature importance plot. Defaults to True
        
    Returns:
        None. Prints results and optionally displays plot
    """
    df = load_transactions(data_path)
    print(f"Loaded {len(df):,} rows from {data_path or 'DEFAULT_DATA_PATH'}")

    engineered_df, X, y, features, _ = engineer_features(df)
    print(f"Total features: {len(features)}")
    print(f"Feature list (first 10): {features[:10]}")
    print("\nClass distribution:")
    print(y.value_counts())

    results = train_and_evaluate(X, y)
    splits = results["splits"]
    print(f"\nTraining set: {len(splits['X_train'])} samples")
    print(f"Test set: {len(splits['X_test'])} samples\n")

    dt_result = results["decision_tree"]
    rf_result = results["random_forest"]

    _print_result(dt_result)
    _print_result(rf_result)

    if show_plot:
        plot_feature_importance(results["feature_importance"])


def _print_result(result) -> None:
    """
    Print formatted model evaluation result.
    
    Args:
        result: ModelResult object with name, accuracy, and report attributes
        
    Returns:
        None. Prints formatted output to stdout
    """
    print("=" * 70)
    print(f"Model: {result.name}")
    print("=" * 70)
    print(f"Accuracy: {result.accuracy:.2%}\n")
    print("Classification Report:")
    print(result.report)
    print()


if __name__ == "__main__":
    main()

