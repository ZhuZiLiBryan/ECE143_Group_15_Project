"""
Optimal Price Recommendation Model for Coffee Sales

Assumptions:
- Input CSV has at least the columns:
    hour_of_day, cash_type, money, coffee_name, Time_of_Day,
    Weekday, Month_name, Weekdaysort, Monthsort, Date, Time
- 'money' is the sale price per unit.
- Cost per drink is approximated via cost_map (edit to your real costs).

This script:
1. Loads data
2. Engineers milk_ratio and aggregated daily data
3. Trains a regression model to predict units sold per day
4. Provides a function to recommend an optimal price for a given coffee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ============================================
# Global plotting style
# ============================================
sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)


# ============================================
# 1. Load Dataset
# ============================================
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ============================================
# 2. Feature: Milk-to-Coffee Ratio
# ============================================
milk_map = {
    "Latte": (0.7, 0.3),
    "Cappuccino": (0.5, 0.5),
    "Flat White": (0.6, 0.4),
    "Hot Chocolate": (1.0, 0.0),
    "Cocoa": (1.0, 0.0),
    "Americano": (0.0, 1.0),
    "Americano with Milk": (0.2, 0.8),
    "Cortado": (0.4, 0.6),
    # Add more drinks or adjust as needed
}


def milk_ratio_from_name(name: str) -> float:
    milk, coffee = milk_map.get(name, (0.0, 1.0))  # default: pure coffee if unknown
    total = milk + coffee
    return milk / total if total > 0 else 0.0


def add_milk_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["milk_ratio"] = df["coffee_name"].apply(milk_ratio_from_name)
    return df


# ============================================
# 3. Aggregate to Daily Coffee-Level Data
# ============================================
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    # Group by coffee_name, Date, and some categorical fields
    daily = (
        df
        .groupby(["coffee_name", "Date", "Weekday", "Month_name", "Time_of_Day"], as_index=False)
        .agg(
            units_sold=("money", "count"),
            avg_price=("money", "mean"),
            avg_milk_ratio=("milk_ratio", "mean"),
        )
    )

    # Create numeric weekday/month indices
    daily["day_of_week_idx"] = pd.Categorical(
        daily["Weekday"],
        categories=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        ordered=True
    ).codes

    daily["month_idx"] = pd.Categorical(
        daily["Month_name"],
        categories=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        ordered=True
    ).codes

    return daily


# ============================================
# 4. Cost per Drink (Approximate — Edit for Reality)
# ============================================
cost_map = {
    "Latte": 20.0,
    "Cappuccino": 18.0,
    "Flat White": 19.0,
    "Hot Chocolate": 15.0,
    "Cocoa": 15.0,
    "Americano": 10.0,
    "Americano with Milk": 12.0,
    "Cortado": 14.0,
    # Add real costs for your drinks
}


def add_costs(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["cost"] = daily["coffee_name"].map(cost_map).fillna(15.0)
    return daily


# ============================================
# 5. Prepare Features & Train Model
# ============================================
def prepare_features(daily: pd.DataFrame):
    target_col = "units_sold"

    feature_cols = [
        "avg_price",
        "avg_milk_ratio",
        "coffee_name",
        "Weekday",
        "Month_name",
        "Time_of_Day",
        "day_of_week_idx",
        "month_idx",
    ]

    X = daily[feature_cols].copy()
    y = daily[target_col].astype(float)

    categorical_features = ["coffee_name", "Weekday", "Month_name", "Time_of_Day"]
    numeric_features = ["avg_price", "avg_milk_ratio", "day_of_week_idx", "month_idx"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    return X, y, preprocessor


def train_demand_model(X: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    regressor = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    print(f"[Model Evaluation] RMSE: {rmse:.3f}, R^2: {r2:.3f}")

    return model


# ============================================
# 6. Optimal Price Recommendation Function
# ============================================
def recommend_optimal_price_for_scenario(
    coffee_name: str,
    daily: pd.DataFrame,
    model: Pipeline,
    weekday: str = "Fri",
    month_name: str = "Mar",
    time_of_day: str = "Morning",
    price_min_factor: float = 0.8,
    price_max_factor: float = 1.2,
    n_grid: int = 30,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Recommend optimal price for a given coffee under a specified scenario.

    Parameters
    ----------
    coffee_name : str
        Name of the coffee drink.
    daily : pd.DataFrame
        Aggregated daily dataset (with avg_price, avg_milk_ratio, cost, etc).
    model : Pipeline
        Trained pipeline (preprocessor + regressor).
    weekday : str
        Day of week scenario (e.g. "Mon", "Tue", ..., "Sun").
    month_name : str
        Month scenario (e.g. "Jan", "Feb", ..., "Dec").
    time_of_day : str
        Time of day scenario (e.g. "Morning", "Afternoon", "Night").
    price_min_factor : float
        Lower bound as a multiple of historical mean price (0.8 = -20%).
    price_max_factor : float
        Upper bound as a multiple of historical mean price (1.2 = +20%).
    n_grid : int
        Number of candidate prices to evaluate.

    Returns
    -------
    results_df : pd.DataFrame
        Dataframe with columns: price, predicted_units, predicted_profit.
    best_row : pd.Series
        Row with the highest predicted_profit.
    """
    base_data = daily[daily["coffee_name"] == coffee_name]
    if base_data.empty:
        raise ValueError(f"No data found for coffee '{coffee_name}'")

    current_price = base_data["avg_price"].mean()
    coffee_cost = base_data["cost"].mean()
    avg_milk_ratio = base_data["avg_milk_ratio"].mean()

    price_min = current_price * price_min_factor
    price_max = current_price * price_max_factor
    price_grid = np.linspace(price_min, price_max, n_grid)

    # Compute indices for chosen weekday/month
    day_idx = pd.Categorical(
        [weekday],
        categories=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        ordered=True
    ).codes[0]

    month_idx = pd.Categorical(
        [month_name],
        categories=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        ordered=True
    ).codes[0]

    candidates = []
    for p in price_grid:
        row = {
            "coffee_name": coffee_name,
            "Weekday": weekday,
            "Month_name": month_name,
            "Time_of_Day": time_of_day,
            "avg_price": p,
            "avg_milk_ratio": avg_milk_ratio,
            "day_of_week_idx": day_idx,
            "month_idx": month_idx,
        }
        candidates.append(row)

    candidates_df = pd.DataFrame(candidates)

    predicted_units = model.predict(candidates_df)
    predicted_profit = (price_grid - coffee_cost) * predicted_units

    results_df = pd.DataFrame({
        "price": price_grid,
        "predicted_units": predicted_units,
        "predicted_profit": predicted_profit,
    })

    best_idx = results_df["predicted_profit"].idxmax()
    best_row = results_df.loc[best_idx]

    return results_df, best_row


# ============================================
# 7. Plotting Helpers
# ============================================
def plot_profit_curve(results_df: pd.DataFrame, best_row: pd.Series, title_suffix: str = ""):
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["price"], results_df["predicted_profit"], marker="o")
    plt.axvline(best_row["price"], color="red", linestyle="--",
                label=f"Optimal price = {best_row['price']:.2f}")
    plt.title(f"Predicted Profit vs Price {title_suffix}")
    plt.xlabel("Price")
    plt.ylabel("Predicted Profit (per day)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_demand_curve(results_df: pd.DataFrame, title_suffix: str = ""):
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["price"], results_df["predicted_units"], marker="o", color="green")
    plt.title(f"Predicted Units Sold vs Price {title_suffix}")
    plt.xlabel("Price")
    plt.ylabel("Predicted Units Sold (per day)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================
# 8. Main Script Example
# ============================================
def main():
    # ---- Step 1: Load raw data ----
    csv_path = "Coffe_sales.csv"  # change to your real path
    df = load_data(csv_path)

    # ---- Step 2: Add milk_ratio ----
    df = add_milk_ratio(df)

    # ---- Step 3: Aggregate to daily coffee-level ----
    daily = aggregate_daily(df)

    # ---- Step 4: Add cost information ----
    daily = add_costs(daily)

    # ---- Step 5: Build features & train model ----
    X, y, preprocessor = prepare_features(daily)
    model = train_demand_model(X, y, preprocessor)

    # ---- Step 6: Example – Optimal price for Latte (Fri Morning, Mar) ----
    coffee_example = "Latte"
    results_latte, best_latte = recommend_optimal_price_for_scenario(
        coffee_name=coffee_example,
        daily=daily,
        model=model,
        weekday="Fri",
        month_name="Mar",
        time_of_day="Morning",
        price_min_factor=0.8,
        price_max_factor=1.2,
        n_grid=40,
    )

    print(f"\nRecommended price for {coffee_example} (Fri Morning, Mar):")
    print(best_latte)

    # ---- Plot curves for this example coffee ----
    plot_profit_curve(results_latte, best_latte,
                      title_suffix=f"for {coffee_example} (Fri Morning, Mar)")
    plot_demand_curve(results_latte,
                      title_suffix=f"for {coffee_example} (Fri Morning, Mar)")

    # ---- Step 7: Recommend prices for all coffees in this scenario ----
    coffee_names = daily["coffee_name"].unique()
    recommendations = []

    for cname in coffee_names:
        try:
            _, best_row = recommend_optimal_price_for_scenario(
                coffee_name=cname,
                daily=daily,
                model=model,
                weekday="Fri",
                month_name="Mar",
                time_of_day="Morning",
                price_min_factor=0.8,
                price_max_factor=1.2,
                n_grid=40,
            )
            recommendations.append({
                "coffee_name": cname,
                "recommended_price": best_row["price"],
                "predicted_units": best_row["predicted_units"],
                "predicted_profit": best_row["predicted_profit"],
            })
        except ValueError:
            continue

    rec_df = pd.DataFrame(recommendations).sort_values("predicted_profit", ascending=False)
    print("\n=== Recommended Prices for All Coffees (Fri Morning, Mar) ===")
    print(rec_df.to_string(index=False))


if __name__ == "__main__":
    main()
