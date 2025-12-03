"""
Visualization utilities for coffee sales analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")


def plot_sales_prediction(daily_sales: pd.Series, moving_avg: pd.Series,
                         forecast: float, next_date: pd.Timestamp,
                         title: str = "Daily Coffee Sales and Next Day Prediction",
                         figsize: tuple = (12, 6)) -> None:
    """
    Plot historical sales, moving average, and prediction.
    
    Args:
        daily_sales: Historical daily sales series
        moving_avg: Moving average series
        forecast: Predicted sales value
        next_date: Date for the prediction
        title: Plot title
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    plt.plot(daily_sales.index, daily_sales.values, lw=0.5, 
             label='Historical Daily Sales')
    plt.plot(moving_avg.index, moving_avg.values, lw=1.5, 
             label='7-Day Moving Average')
    
    # Plot prediction
    plt.scatter(next_date, forecast, color='red', 
               label='Predicted Next Day', zorder=5)
    plt.plot([daily_sales.index[-1], next_date], 
             [daily_sales.iloc[-1], forecast], 
             'r--', lw=1, label='Trend to Prediction')
    
    # Annotate prediction
    plt.annotate(f"{forecast:.2f}", (next_date, forecast), 
                textcoords="offset points", xytext=(0, 10), 
                ha='center', color='red', fontsize=10)
    
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales (money)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_coffee_predictions(predictions: dict, title: str = "Predicted Coffee Sales by Type",
                           figsize: tuple = (8, 5)) -> None:
    """
    Plot bar chart of predicted sales for each coffee type.
    
    Args:
        predictions: Dictionary mapping coffee names to predicted sales
        title: Plot title
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    plt.bar(predictions.keys(), predictions.values(), color='skyblue')
    plt.xlabel('Coffee Name')
    plt.ylabel('Predicted Sales Next Month')
    plt.title(title)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()


def plot_promotion_frequency(promotion_recommendations: pd.DataFrame,
                            title: str = "Frequency of Drink Promotions by Month",
                            figsize: tuple = (12, 6)) -> None:
    """
    Plot monthly frequency of promotion recommendations.
    
    Args:
        promotion_recommendations: DataFrame with dates as index and 'recommended_drink' column
        title: Plot title
        figsize: Figure size tuple
    """
    df = promotion_recommendations.copy()
    df.index = pd.to_datetime(df.index)
    
    # Group by month and drink
    per_month = (
        df.assign(month=df.index.to_period('M'))
        .groupby(['month', 'recommended_drink'])
        .size()
        .unstack(fill_value=0)
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    per_month.plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel("Number of Recommendations")
    ax.set_xlabel("Month")
    ax.set_title(title)
    ax.legend(title="Drink", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_scenario_analysis(impact_results: dict, profit_margin_scenarios: dict,
                          rolling_windows: list,
                          title: str = "Impact of Changing Profit Margins & Rolling Windows",
                          figsize: tuple = (12, 8)) -> None:
    """
    Plot promotion recommendations under different scenarios.
    
    Args:
        impact_results: Dictionary mapping scenario keys to recommendation series
        profit_margin_scenarios: Dictionary of scenario names
        rolling_windows: List of rolling window sizes
        title: Plot title
        figsize: Figure size tuple
    """
    fig, axs = plt.subplots(len(profit_margin_scenarios), len(rolling_windows), 
                            figsize=figsize, sharey=True)
    
    for i, (margin_label, _) in enumerate(profit_margin_scenarios.items()):
        for j, window in enumerate(rolling_windows):
            key = f"{margin_label}_profit_margin__{window}d_rolling"
            series = impact_results[key]
            df = pd.DataFrame({'recommended_drink': series})
            df.index = pd.to_datetime(df.index)
            
            per_month = (
                df.assign(month=df.index.to_period('M'))
                .groupby(['month', 'recommended_drink'])
                .size()
                .unstack(fill_value=0)
            )
            
            ax = axs[i, j]
            per_month.plot(kind='bar', stacked=True, ax=ax, legend=False)
            # Add legend only for first subplot so we can use its handles/labels later
            if i == 0 and j == 0:
                handles, labels = ax.get_legend_handles_labels()
            ax.set_title(f"{margin_label.capitalize()} profit, {window}d rolling")
            if j == 0:
                ax.set_ylabel("Monthly Recommendations")
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")

    # Adjust figure size/spacing to ensure the plot fits within the output window
    plt.subplots_adjust(left=0.08, right=0.88, top=0.9, bottom=0.15, wspace=0.3, hspace=0.4)
    axs[0, 0].legend(handles, labels, title="Drink", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    for j, window in enumerate(rolling_windows):
        axs[0, j].set_xlabel(f"{window}-day Rolling Window")
    
    plt.tight_layout()
    plt.legend(title="Drink", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.show()

