import pandas as pd
import os

try:
    from .kmeans import create_rfm_features, load_and_preprocess_data, perform_clustering, plot_elbow_method
except ImportError:
    from kmeans import create_rfm_features, load_and_preprocess_data, perform_clustering, plot_elbow_method

def kmeans_main(data_path: str = None):
    '''
    Main execution for K-Means Clustering Analysis.

    Args:
        data_path: Path to the CSV file (relative to project root). 
                   If None, uses default path 'index_1.csv'.
    
    Returns:
    None
    '''
    
    # Use data path if not provided
    if data_path is None:
        data_path = 'upload/index_1.csv'

    print(f"Target data file: {data_path}")

    # Load and Preprocess Data
    df = load_and_preprocess_data(data_path)
    
    if df is not None:
        # RFM Feature Engineering
        rfm_df, X_scaled, features = create_rfm_features(df)
        
        # Elbow Method (Visualize to choose k)
        plot_elbow_method(X_scaled)
        
        # Final Clustering (Assuming k=3 based on Elbow plot)
        perform_clustering(rfm_df, X_scaled, features, k=3)

if __name__ == "__main__":
    kmeans_main()