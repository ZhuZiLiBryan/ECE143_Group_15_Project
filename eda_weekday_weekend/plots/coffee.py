"""Popular coffee comparison visualization."""
from __future__ import annotations

import matplotlib.pyplot as plt

from ..config import DAY_TYPE_ORDER, DAY_TYPE_COLOR_MAP, FIG_SIZE_TRIPLE
from ..data_loader import load_and_preprocess
from .style import init_style

init_style()


def eda_popular_coffee_comparison(data_path: str | None = None) -> None:
    """
    Display Top 5 popular coffees comparison by day type as horizontal bar charts.
    
    Args:
        data_path: Optional path to CSV file. If None, uses default path
        
    Returns:
        None. Displays the plot using plt.show()
    """
    df = load_and_preprocess(data_path)

    coffee_stats = df.groupby(["day_type", "coffee_name"]).size().reset_index(name="count")
    total_by_day = df.groupby("day_type").size().reset_index(name="total")
    coffee_stats = coffee_stats.merge(total_by_day, on="day_type")
    coffee_stats["percentage"] = (coffee_stats["count"] / coffee_stats["total"] * 100).round(2)

    top_coffees = {}
    for day_type in DAY_TYPE_ORDER:
        data = (
            coffee_stats[coffee_stats["day_type"] == day_type]
            .sort_values("count", ascending=False)
            .head(5)
        )
        top_coffees[day_type] = data

    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE_TRIPLE)

    for idx, day_type in enumerate(DAY_TYPE_ORDER):
        data = top_coffees[day_type].sort_values("percentage", ascending=True)
        bars = axes[idx].barh(
            data["coffee_name"],
            data["percentage"],
            color=DAY_TYPE_COLOR_MAP[day_type],
            alpha=0.8,
        )
        axes[idx].set_title(f"{day_type}", fontsize=14, fontweight="bold", pad=15)
        axes[idx].set_xlabel("Percentage (%)", fontsize=11)
        axes[idx].grid(axis="x", alpha=0.3, linestyle="--")
        axes[idx].set_xlim(0, max(data["percentage"]) * 1.15)

        for bar, pct in zip(bars, data["percentage"]):
            axes[idx].text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2.0,
                f"{pct:.1f}%",
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        if idx == 0:
            axes[idx].set_ylabel("Coffee Type", fontsize=11)

    plt.suptitle(
        "Top 5 Popular Coffees Comparison by Day Type",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

