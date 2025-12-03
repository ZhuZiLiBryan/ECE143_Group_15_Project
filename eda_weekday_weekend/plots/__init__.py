"""Plotting sub-package for weekday/weekend EDA."""
from __future__ import annotations

from .sales import eda_sales_comparison
from .coffee import eda_popular_coffee_comparison
from .order_value import eda_order_value_statistics
from .style import init_style


def run_all_eda(data_path: str | None = None) -> None:
    """
    Execute all EDA visualizations in sequence.
    
    Args:
        data_path: Optional path to CSV file. If None, uses default path
        
    Returns:
        None. Displays all plots sequentially
    """
    eda_sales_comparison(data_path)
    eda_popular_coffee_comparison(data_path)
    eda_order_value_statistics(data_path)


__all__ = [
    "eda_sales_comparison",
    "eda_popular_coffee_comparison",
    "eda_order_value_statistics",
    "run_all_eda",
    "init_style",
]

