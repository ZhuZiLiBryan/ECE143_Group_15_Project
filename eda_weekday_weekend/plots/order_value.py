"""Order value statistics visualization."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from ..config import DAY_TYPE_ORDER, DAY_TYPE_COLORS, FIG_SIZE_WIDE
from ..data_loader import load_and_preprocess
from .style import init_style

init_style()


def eda_order_value_statistics(data_path: str | None = None) -> None:
    """
    Display average order value comparison and distribution box plot by day type.
    
    Args:
        data_path: Optional path to CSV file. If None, uses default path
        
    Returns:
        None. Displays the plot using plt.show()
    """
    df = load_and_preprocess(data_path)

    avg_order_stats = df.groupby("day_type")["money"].agg(
        [("Mean", "mean"), ("Median", "median"), ("Std Dev", "std"), ("Min", "min"), ("Max", "max")]
    ).round(2)

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_WIDE)

    # Average order value bar chart
    avg_prices = avg_order_stats.loc[DAY_TYPE_ORDER, "Mean"]
    bars = axes[0].bar(DAY_TYPE_ORDER, avg_prices, color=DAY_TYPE_COLORS, width=0.6, alpha=0.8)
    axes[0].set_title("Average Order Value Comparison", fontsize=16, fontweight="bold", pad=20)
    axes[0].set_ylabel("Average Order Value ($)", fontsize=12)
    axes[0].set_xlabel("Day Type", fontsize=12)
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")
    axes[0].set_ylim(0, max(avg_prices) * 1.2)
    for bar, value in zip(bars, avg_prices):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"${value:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    # Box plot
    sns.boxplot(
        data=df,
        x="day_type",
        y="money",
        order=DAY_TYPE_ORDER,
        palette=DAY_TYPE_COLORS,
        ax=axes[1],
        width=0.5,
    )
    axes[1].set_title("Order Value Distribution", fontsize=16, fontweight="bold", pad=20)
    axes[1].set_ylabel("Order Amount ($)", fontsize=12)
    axes[1].set_xlabel("Day Type", fontsize=12)
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.show()

