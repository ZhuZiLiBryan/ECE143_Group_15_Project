import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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


def create_rfm_features(df):
    # Filter for card transactions only (as they have unique IDs)
    card_df = df[df['cash_type'] == 'card'].dropna(subset=['card']).copy()
    max_date = card_df['datetime'].max()

    # Aggregate data by card ID
    customer_summary = card_df.groupby('card').agg(
        total_visits=pd.NamedAgg(column='datetime', aggfunc='count'), # Frequency
        total_spent=pd.NamedAgg(column='money', aggfunc='sum'),       # Monetary
        last_visit=pd.NamedAgg(column='datetime', aggfunc='max')      # Used for Recency
    ).reset_index()

    # Calculate Recency (days since the last visit)
    customer_summary['days_since_last_visit'] = (max_date - customer_summary['last_visit']).dt.days

    # Select specific features for clustering
    rfm_features = ['days_since_last_visit', 'total_visits', 'total_spent']
    X = customer_summary[rfm_features]

    # Scale the data (StandardScaler is crucial for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return customer_summary, X_scaled, rfm_features

def plot_elbow_method(X_scaled, max_k=10):
    inertia = {}
    
    # Iterate through possible k values
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        kmeans.fit(X_scaled)
        inertia[k] = kmeans.inertia_

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(list(inertia.keys()), list(inertia.values()), marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.show()

def perform_clustering(customer_summary, X_scaled, rfm_features, k=3):   
    # Initialize and fit the final model
    kmeans_final = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans_final.fit(X_scaled)

    # Assign cluster labels back to the original dataframe
    customer_summary['cluster'] = kmeans_final.labels_
    
    # Calculate mean values for each cluster to interpret them
    cluster_analysis = customer_summary.groupby('cluster')[rfm_features].mean()
    
    print("\nCluster Analysis (Mean Values):")
    print(cluster_analysis)
    
    print("\nCustomer Counts per Cluster:")
    print(customer_summary['cluster'].value_counts().sort_index())
    
    return customer_summary

