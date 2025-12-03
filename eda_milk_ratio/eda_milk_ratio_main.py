import pandas as pd

from .eda_milk_ratio_deps.milk_ratio_calculations import determine_milk_ratio
from .eda_milk_ratio_deps.milk_ratio_scatterplot import milk_ratio_scatter
from .eda_milk_ratio_deps.milk_ratio_heatmap import milk_ratio_heatmap

def eda_milk_main(data_path: str = None):
    '''
    Main execution for milk ratio EDA.

    Args:
        data_path: Path to the CSV file (relative to project root). 
                    If None, uses path from config.json
    
    Returns:
    None
    '''

    # Use data path from config if not provided
    if data_path is None:
        data_path = 'Coffe_sales.csv'

    df = pd.read_csv(data_path)

    # Calculate milk ratios and append to dataframe
    df["milk_ratio"] = df["coffee_name"].apply(determine_milk_ratio)

    # Calculate and create scatterplot for milk
    print("--------------------")
    print("Plotting Average Milk Ratio by Hour")
    print("--------------------")
    milk_ratio_scatter(df)

    # Calculate and create heatmap for milk 
    print("--------------------")
    print("Plotting Average Milk Ratio Heatmap")
    print("--------------------")
    milk_ratio_heatmap(df)

if __name__ == "__main__":
    eda_milk_main()