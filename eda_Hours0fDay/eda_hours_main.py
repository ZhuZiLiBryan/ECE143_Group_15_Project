import os

try:
    from .eda_hoursOfDay import load_and_preprocess_data, plot_transactions_by_hour
except ImportError:
    from eda_hoursOfDay import load_and_preprocess_data, plot_transactions_by_hour

def eda_hours_main(data_path: str = None):
    '''
    Main execution for Hourly Transactions EDA.

    Args:
        data_path: Path to the CSV file (relative to project root). 
                   If None, uses default path 'index_1.csv'.
    
    Returns:
    None
    '''
    
    if data_path is None:
        data_path = 'index_1.csv'

    print(f"Target data file: {data_path}")

    # Load and Preprocess Data
    df = load_and_preprocess_data(data_path)
    
    if df is not None:
        # Plot Transactions
        plot_transactions_by_hour(df)

if __name__ == "__main__":
    eda_hours_main()