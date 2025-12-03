"""Weekday/Weekend/Holiday EDA package extracted from weekday_weekend_analysis.ipynb."""

from .plots import (
    eda_sales_comparison,
    eda_popular_coffee_comparison,
    eda_order_value_statistics,
    run_all_eda,
)

# Alias for consistent naming convention across modules
eda_weekday_weekend_main = run_all_eda

__all__ = [
    "eda_sales_comparison",
    "eda_popular_coffee_comparison",
    "eda_order_value_statistics",
    "run_all_eda",
    "eda_weekday_weekend_main",
]

