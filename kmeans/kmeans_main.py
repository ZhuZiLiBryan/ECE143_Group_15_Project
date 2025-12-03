# --- Main Execution Block ---
from kmeans import create_rfm_features, load_and_preprocess_data, perform_clustering, plot_elbow_method


if __name__ == "__main__":
    file_path = 'index_1.csv'
    
    #  Load Data
    df = load_and_preprocess_data(file_path)
    
    if df is not None:
        #  RFM Feature Engineering
        rfm_df, X_scaled, features = create_rfm_features(df)
        
        #  Elbow Method (Visualize to choose k)
        plot_elbow_method(X_scaled)
        
        #  Final Clustering (Assuming k=3 based on Elbow plot)
        perform_clustering(rfm_df, X_scaled, features, k=3)