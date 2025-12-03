# ECE143_Group_15_Project
Project for ECE143, Data Visualization


## How to Run the Sales and Promotion Analysis

To execute the coffee sales and promotion analysis, run the following command from the project root directory:

```
python run_analysis.py
```

This will process the sales dataset, generate predictions, and provide visualizations and recommendations based on the most recent data. Ensure all required dependencies are installed before running the script.


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
