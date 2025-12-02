"""
Coffee sales analysis scripts package.
"""

from .data_loader import (
    load_data,
    preprocess_datetime,
    normalize_coffee_names,
    prepare_daily_sales,
    prepare_daily_coffee_sales
)
from .sales_prediction import (
    predict_next_day_sales,
    calculate_moving_average
)
from .coffee_prediction import predict_most_sold_coffee_month
from .promotion_recommendation import (
    get_default_profit_margins,
    recommend_daily_promotions,
    analyze_promotion_scenarios
)
from .visualization import (
    plot_sales_prediction,
    plot_coffee_predictions,
    plot_promotion_frequency,
    plot_scenario_analysis
)
from .config_loader import load_config, get_profit_margins

__all__ = [
    'load_data',
    'preprocess_datetime',
    'normalize_coffee_names',
    'prepare_daily_sales',
    'prepare_daily_coffee_sales',
    'predict_next_day_sales',
    'calculate_moving_average',
    'predict_most_sold_coffee_month',
    'load_config',
    'get_profit_margins',
    'get_default_profit_margins',
    'recommend_daily_promotions',
    'analyze_promotion_scenarios',
    'plot_sales_prediction',
    'plot_coffee_predictions',
    'plot_promotion_frequency',
    'plot_scenario_analysis',
]

