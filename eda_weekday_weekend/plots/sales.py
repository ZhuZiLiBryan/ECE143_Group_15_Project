"""Sales comparison visualization."""
from __future__ import annotations

import matplotlib.pyplot as plt

from ..config import DAY_TYPE_ORDER, DAY_TYPE_COLORS, FIG_SIZE_WIDE
from ..data_loader import load_and_preprocess, compute_daily_sales
from .style import init_style

init_style()


def eda_sales_comparison(data_path: str | None = None) -> None:
    """
    Display average daily sales and order count comparison by day type.
    
    Args:
        data_path: Optional path to CSV file. If None, uses default path
        
    Returns:
        None. Displays the plot using plt.show()
    """
    df = load_and_preprocess(data_path)
    daily_sales = compute_daily_sales(df)

    avg_stats = daily_sales.groupby("day_type").agg(
        {"total_sales": "mean", "order_count": "mean"}
    ).round(2)

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_WIDE)

    # Average daily sales
    avg_sales = avg_stats.loc[DAY_TYPE_ORDER, "total_sales"]
    bars1 = axes[0].bar(DAY_TYPE_ORDER, avg_sales, color=DAY_TYPE_COLORS, width=0.6)
    axes[0].set_title("Average Daily Sales Comparison", fontsize=16, fontweight="bold", pad=20)
    axes[0].set_ylabel("Average Sales ($)", fontsize=12)
    axes[0].set_xlabel("Day Type", fontsize=12)
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")
    axes[0].set_ylim(0, max(avg_sales) * 1.2)
    for bar, value in zip(bars1, avg_sales):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"${value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    # Average daily order count
    avg_orders = avg_stats.loc[DAY_TYPE_ORDER, "order_count"]
    bars2 = axes[1].bar(DAY_TYPE_ORDER, avg_orders, color=DAY_TYPE_COLORS, width=0.6)
    axes[1].set_title("Average Daily Order Count Comparison", fontsize=16, fontweight="bold", pad=20)
    axes[1].set_ylabel("Average Order Count", fontsize=12)
    axes[1].set_xlabel("Day Type", fontsize=12)
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")
    axes[1].set_ylim(0, max(avg_orders) * 1.2)
    for bar, value in zip(bars2, avg_orders):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    plt.tight_layout()
    plt.show()

