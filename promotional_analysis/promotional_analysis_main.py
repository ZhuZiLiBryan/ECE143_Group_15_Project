"""
Main script for coffee sales analysis and predictions.
"""

import pandas as pd

# Import modules 
try:
    from .data_loader import (
        load_data, preprocess_datetime, normalize_coffee_names,
        prepare_daily_sales, prepare_daily_coffee_sales
    )
    from .sales_prediction import predict_next_day_sales, calculate_moving_average
    from .coffee_prediction import predict_most_sold_coffee_month, predict_most_sold_coffee_week
    from .promotion_recommendation import (
        get_default_profit_margins, recommend_daily_promotions,
        analyze_promotion_scenarios
    )
    from .visualization import (
        plot_sales_prediction, plot_coffee_predictions,
        plot_promotion_frequency, plot_scenario_analysis
    )
    from .config_loader import load_config
except ImportError:
    # Fallback for direct execution
    from data_loader import (
        load_data, preprocess_datetime, normalize_coffee_names,
        prepare_daily_sales, prepare_daily_coffee_sales
    )
    from sales_prediction import predict_next_day_sales, calculate_moving_average
    from coffee_prediction import predict_most_sold_coffee_month
    from .promotion_recommendation import (
        get_default_profit_margins, recommend_daily_promotions,
        analyze_promotion_scenarios
    )
    from visualization import (
        plot_sales_prediction, plot_coffee_predictions,
        plot_promotion_frequency, plot_scenario_analysis
    )
    from config_loader import load_config


def promotional_analsysis_main(data_path: str = None, config_path: str = None):
    """
    Main execution function.
    
    Args:
        data_path: Path to the CSV file (relative to project root). 
                  If None, uses path from config.json
        config_path: Path to config file. If None, uses default config.json
    """
    # Load configuration
    config = load_config(config_path)
    
    # Use data path from config if not provided
    if data_path is None:
        data_path = config.get('data_path', 'upload/index_1.csv')
    
    # Load and preprocess data
    print("Loading data...")
    df = load_data(data_path)
    df = preprocess_datetime(df)
    df = normalize_coffee_names(df)
    
    print(f"Data loaded: {len(df)} records")
    print(f"Coffee types: {df['new_coffee_name'].value_counts().to_dict()}\n")
    
    # Get settings from config
    sales_config = config.get('sales_prediction', {})
    coffee_config = config.get('coffee_prediction', {})
    promotion_config = config.get('promotion', {})
    scenario_config = config.get('scenario_analysis', {})
    
    # 1. Predict next day sales
    print("=" * 60)
    print("1. Sales Prediction")
    print("=" * 60)
    daily_sales = prepare_daily_sales(df)
    forecast = predict_next_day_sales(
        daily_sales,
        training_days=sales_config.get('training_days', 365),
        order=tuple(sales_config.get('order', [1, 1, 1])),
        seasonal_order=tuple(sales_config.get('seasonal_order', [1, 1, 1, 12]))
    )
    moving_avg = calculate_moving_average(daily_sales, window=7)
    next_date = daily_sales.index[-1] + pd.offsets.Day(1)
    
    print(f"Predicted coffee sales for next day: {forecast:.2f}")
    plot_sales_prediction(daily_sales, moving_avg, forecast, next_date)
    
    # 2. Predict most sold coffee
    print("\n" + "=" * 60)
    print("2. Most Popular Coffee Prediction")
    print("=" * 60)
    most_sold, predicted_sales, all_predictions = predict_most_sold_coffee_month(
        df,
        months_back=coffee_config.get('months_back', 12),
        order=tuple(coffee_config.get('order', [1, 1, 1])),
        seasonal_order=tuple(coffee_config.get('seasonal_order', [1, 1, 1, 12]))
    )
    print(f"Predicted most sold coffee for next month: {most_sold}")
    print(f"Predicted sales: {predicted_sales:.2f}")
    plot_coffee_predictions(all_predictions)
    
    most_sold, predicted_sales, all_predictions = predict_most_sold_coffee_week(
        df,
        weeks_back=coffee_config.get('weeks_back', 4),
        order=tuple(coffee_config.get('order', [1, 1, 1])),
        seasonal_order=tuple(coffee_config.get('seasonal_order', [1, 1, 1, 12]))
    )
    print(f"Predicted most sold coffee for next week: {most_sold}")
    print(f"Predicted sales: {predicted_sales:.2f}")
    plot_coffee_predictions(all_predictions)
    
    # 3. Promotion recommendations
    print("\n" + "=" * 60)
    print("3. Daily Promotion Recommendations")
    print("=" * 60)
    daily_coffee_sales = prepare_daily_coffee_sales(df)
    profit_margins = get_default_profit_margins(
        daily_coffee_sales.columns.tolist(), 
        config=config
    )
    default_margin = config.get('profit_margins', {}).get('default', 2.0)
    promotion_recommendations = recommend_daily_promotions(
        daily_coffee_sales, 
        profit_margins, 
        rolling_window=promotion_config.get('rolling_window', 7),
        default_margin=default_margin
    )
    
    print("Recommended promotion drink per day (top 7 latest):")
    for day in list(promotion_recommendations.index)[-7:]:
        print(f"{day.date()}: Promote '{promotion_recommendations[day]}'")
    
    promotion_df = pd.DataFrame({'recommended_drink': promotion_recommendations})
    print("\nLast 14 days recommendations:")
    print(promotion_df.tail(14))
    
    plot_promotion_frequency(promotion_df)
    
    # 4. Scenario analysis
    print("\n" + "=" * 60)
    print("4. Scenario Analysis")
    print("=" * 60)
    multipliers = scenario_config.get('profit_multipliers', {
        'base': 1.0,
        'lowered': 0.8,
        'raised': 1.2
    })
    profit_margin_scenarios = {
        'base': profit_margins,
        'lowered': {k: v * multipliers.get('lowered', 0.8) for k, v in profit_margins.items()},
        'raised': {k: v * multipliers.get('raised', 1.2) for k, v in profit_margins.items()},
    }
    rolling_windows = scenario_config.get('rolling_windows', [3, 7, 14, 30])
    
    impact_results = analyze_promotion_scenarios(
        df, profit_margin_scenarios, rolling_windows
    )
    
    print("Example: Most common recommendation under different scenarios (last 12 months)")
    for key, series in impact_results.items():
        month_recommend = pd.DataFrame({'recommended_drink': series})
        top_recommendations = month_recommend['recommended_drink'].value_counts().head(3)
        print(f"\n{key}:")
        print(top_recommendations)
        print('-' * 40)
    
    plot_scenario_analysis(impact_results, profit_margin_scenarios, rolling_windows)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

    # To run analysis, execute the following command from the project root directory:
    # 
    #     python run_analysis.py
    # 
