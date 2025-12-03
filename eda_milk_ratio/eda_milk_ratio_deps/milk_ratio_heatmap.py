import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def milk_ratio_heatmap(df):
    '''
    Plot a heatmap of avg milk ratio by the hour of day.

    Args:
    df : pd.DataFrame
        Dataframe of values
    
    Returns:
    None
    '''
    # heatmap
    df["ratio_bucket"] = pd.cut(df["milk_ratio"], bins=[0,0.25,0.5,0.75,1.0])

    heat_groups = df.groupby(["hour_of_day", "ratio_bucket"], observed=False)
    heats = heat_groups.size().unstack(fill_value=0)

    plt.figure()
    sns.heatmap(heats, cmap="YlOrBr")
    plt.title("Milk Ratio to Sales Heatmap (by Hour)")
    plt.xlabel("Milk Ratio")
    plt.ylabel("Hour of Day")
    plt.show()
