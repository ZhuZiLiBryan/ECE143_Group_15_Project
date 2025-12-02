"""
Promotion recommendation system for maximizing profit.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from .config_loader import get_profit_margins, load_config


def get_default_profit_margins(coffee_names: list, 
                               config: Optional[Dict] = None) -> Dict[str, float]:
    """
    Get profit margins for coffee types from config file.
    
    Args:
        coffee_names: List of unique coffee names
        config: Optional configuration dictionary. If None, loads from config.json
        
    Returns:
        Dictionary mapping coffee names to profit margins
    """
    return get_profit_margins(config=config, coffee_names=coffee_names)


def recommend_daily_promotions(daily_coffee_sales: pd.DataFrame,
                              profit_margins: Dict[str, float],
                              rolling_window: int = 7,
                              default_margin: float = 2.0) -> pd.Series:
    """
    Recommend which drink to promote each day to maximize profit.
    
    Args:
        daily_coffee_sales: DataFrame with dates as index and coffee types as columns
        profit_margins: Dictionary mapping coffee names to profit margins
        rolling_window: Window size for calculating rolling averages
        default_margin: Default profit margin for drinks not in profit_margins
        
    Returns:
        Series with recommended drink for each day
    """
    # Calculate rolling average sales for each drink
    rolling_avg = daily_coffee_sales.rolling(window=rolling_window, min_periods=1).mean()
    
    promotion_targets = {}
    
    for day in daily_coffee_sales.index:
        today_sales = daily_coffee_sales.loc[day]
        trend = rolling_avg.loc[day]
        
        # Calculate gap between trend and today's sales
        gap = trend.max() - today_sales
        
        # Score each drink: gap * profit margin
        score = gap * today_sales.index.map(lambda x: profit_margins.get(x, default_margin))
        
        # If all drinks are at trend, use profit margin alone
        if (score <= 0).all():
            score = today_sales.index.map(lambda x: profit_margins.get(x, default_margin))
        
        # Select drink with highest score
        drink_to_promote = score.idxmax()
        promotion_targets[day] = drink_to_promote
    
    return pd.Series(promotion_targets)


def analyze_promotion_scenarios(df: pd.DataFrame, 
                                profit_margin_scenarios: Dict[str, Dict[str, float]],
                                rolling_windows: list,
                                date_col: str = 'date',
                                coffee_col: str = 'new_coffee_name') -> Dict[str, pd.Series]:
    """
    Analyze promotion recommendations under different profit margin scenarios.
    
    Args:
        df: Input DataFrame with sales data
        profit_margin_scenarios: Dictionary of scenario names to profit margin dicts
        rolling_windows: List of rolling window sizes to test
        date_col: Name of the date column
        coffee_col: Name of the coffee name column
        
    Returns:
        Dictionary mapping scenario keys to recommendation series
    """
    sales_df = df.copy()
    sales_df[date_col] = pd.to_datetime(sales_df[date_col])
    sales_df = sales_df.set_index(date_col).sort_index()
    
    impact_results = {}
    
    for margin_label, margin_dict in profit_margin_scenarios.items():
        for window in rolling_windows:
            # Calculate daily profit per drink
            daily_profits = sales_df.copy()
            daily_profits['profit'] = daily_profits[coffee_col].map(margin_dict)
            
            if window > 1:
                profit_per_day_drink = (
                    daily_profits.groupby([pd.Grouper(freq='D'), coffee_col])['profit']
                    .sum()
                    .unstack(fill_value=0)
                    .rolling(window=window, min_periods=1)
                    .sum()
                )
            else:
                profit_per_day_drink = (
                    daily_profits.groupby([pd.Grouper(freq='D'), coffee_col])['profit']
                    .sum()
                    .unstack(fill_value=0)
                )
            
            # Recommend drink with highest rolling profit
            rolling_targets = {}
            for day in profit_per_day_drink.index:
                most_profitable = profit_per_day_drink.loc[day].idxmax()
                rolling_targets[day] = most_profitable
            
            key = f"{margin_label}_profit_margin__{window}d_rolling"
            impact_results[key] = pd.Series(rolling_targets)
    
    return impact_results

