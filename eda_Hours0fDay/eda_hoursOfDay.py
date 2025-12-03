import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set global plot style
sns.set_style("whitegrid")

def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

    # Convert columns to datetime objects
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = pd.to_datetime(df['date'])

    # Feature Engineering: Extract time components
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week_num'] = df['datetime'].dt.dayofweek
    df['month_num'] = df['datetime'].dt.month
    df['day_of_week_name'] = df['datetime'].dt.day_name()
    # Create binary weekend flag (1 if weekend, 0 if not)
    df['is_weekend'] = df['day_of_week_name'].isin(['Saturday', 'Sunday']).astype(int)
    
    print("Data loaded and preprocessed successfully.")
    return df

def plot_transactions_by_hour(df):
    plt.figure(figsize=(12, 6))
    
    # Create count plot using seaborn
    sns.countplot(x='hour', data=df, palette='coolwarm')
    
    # Set titles and labels
    plt.title('Number of Transactions by Hour of Day') 
    plt.xlabel('Hour of Day (0-23)') 
    plt.ylabel('Number of Transactions') 
    
    # Display the plot
    plt.show()

