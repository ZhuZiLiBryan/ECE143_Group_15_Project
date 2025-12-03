# ECE143_Group_15_Project
Project for ECE143, Data Visualization.  A link to a recorded presentation can be found [here](https://www.youtube.com/watch?v=Bdt3se5QOy8&feature=youtu.be).

## How to Run All Analyses/Visualizations 

First ensure all dependencies are installed through `requirements.txt`.
```
pip install -r requirements.txt
```

Next, to execute the all components of the analysis, run the following command from the project root directory:

```
python run_analysis.py
```

Observe the following sections for more granular control of specific visualizations/analysis.

## File Structure:

## Third Party Dependencies:
```
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scipy
- scikit-learn
- holidays
```

## File Structure
```
ECE143_Group_15_Project/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── config.json                        # Configuration file
├── run_analysis.py                    # Main entry point to run all analyses
│
├── all_visualizations.ipynb          # Jupyter notebook for all visualizations
│
├── eda_Hours0fDay/                   # Hour of day analysis
│   ├── __init__.py
│   ├── eda_hours_main.py             # Main entry point
│   └── eda_hoursOfDay.py             # Hourly transaction analysis
│
├── eda_milk_ratio/                   # Milk ratio exploratory data analysis
│   ├── __pycache__/
│   ├── eda_milk_ratio_main.py        # Main entry point
│   └── eda_milk_ratio_deps/          # Milk ratio dependencies
│       ├── __init__.py
│       ├── __pycache__/
│       ├── milk_ratio_calculations.py # Milk ratio calculations
│       ├── milk_ratio_heatmap.py     # Heatmap visualization
│       └── milk_ratio_scatterplot.py # Scatter plot visualization
│
├── eda_weekday_weekend/              # Weekday/Weekend/Holiday analysis
│   ├── __init__.py
│   ├── __pycache__/
│   ├── config.py                     # Configuration settings
│   ├── data_loader.py                # Data loading and preprocessing
│   └── plots/                        # Visualization modules
│       ├── __init__.py
│       ├── __pycache__/
│       ├── coffee.py                 # Coffee popularity plots
│       ├── order_value.py            # Order value statistics
│       ├── sales.py                  # Sales comparison plots
│       └── style.py                  # Plot styling utilities
│
├── kmeans/                           # K-Means clustering analysis
│   ├── kmeans_main.py                # Main entry point
│   └── kmeans.py                     # K-Means clustering implementation
│
├── promotional_analysis/             # Promotional analysis and predictions
│   ├── __init__.py
│   ├── __pycache__/
│   ├── promotional_analysis_main.py  # Main entry point
│   ├── config_loader.py              # Configuration loader
│   ├── data_loader.py                # Data loading utilities
│   ├── coffee_prediction.py          # Coffee sales prediction
│   ├── sales_prediction.py           # Sales forecasting (SARIMAX)
│   ├── promotion_recommendation.py   # Promotion recommendation engine
│   └── visualization.py              # Visualization utilities
│
├── user_analysis/                    # User behavior analysis and prediction
│   ├── __init__.py
│   ├── __pycache__/
│   ├── main.py                       # Main pipeline entry point
│   ├── config.py                     # Configuration settings
│   ├── data_loader.py                # Data loading utilities
│   ├── visualization.py              # Feature importance visualization
│   ├── features/                     # Feature engineering modules
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   ├── temporal.py               # Temporal feature engineering
│   │   ├── customer.py               # Customer history features
│   │   ├── price.py                  # Price-related features
│   │   └── encoding.py               # Categorical encoding
│   └── models/                       # Machine learning models
│       ├── __init__.py
│       ├── __pycache__/
│       ├── decision_tree.py          # Decision Tree classifier
│       ├── random_forest.py          # Random Forest classifier
│       └── evaluation.py             # Model evaluation metrics
│
└── upload/                           # Data and output files
    └── index_1.csv                   # Main dataset
```

## Milk Ratio EDA
From within `/eda_mik_ratio`, you can run the following command.

```bash
python -m eda_milk_ratio_main.py
```

## Hourly transaction EDA
From within `/eda_HoursOfDay`, you can run the following command.

```bash
python -m eda_hours_main.py
```
## Kmeans clustering
From within `/kmenas`, you can run the following command.

```bash
python -m kmeans_main.py
```

## user_analysis

Coffee type prediction pipeline extracted from `upload/model_user.ipynb`.

```python
from user_analysis import show_feature_importance

# Display Random Forest feature importance chart
show_feature_importance()
```

Run the full pipeline from command line:

```bash
python -m user_analysis.main
```

Structure:
- `features/` – temporal, customer, price & encoding feature engineering
- `models/` – Decision Tree & Random Forest training and evaluation
- `main.py` – end-to-end pipeline entry point

## weekday_weekend_eda

Weekday/Weekend/Holiday comparative EDA extracted from `weekday_weekend_analysis.ipynb`.

```python
from weekday_weekend_eda import (
    eda_sales_comparison,
    eda_popular_coffee_comparison,
    eda_order_value_statistics,
    run_all_eda,
)

# Run all EDA visualizations
run_all_eda()
```

Structure:
- `plots/` – sales, coffee popularity, order value visualizations
- `data_loader.py` – load data and classify day types (Weekday/Weekend/Holiday)
