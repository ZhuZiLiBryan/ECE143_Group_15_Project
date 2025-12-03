from eda_hoursOfDay import load_and_preprocess_data, plot_transactions_by_hour


if __name__ == "__main__":
    file_path = 'index_1.csv'
    
    df = load_and_preprocess_data(file_path)
    
    if df is not None:
        plot_transactions_by_hour(df)