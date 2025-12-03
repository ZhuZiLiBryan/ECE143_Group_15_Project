import matplotlib.pyplot as plt
import seaborn as sns

def milk_ratio_scatter(df):
    '''
    Plot a scatterplot of avg milk ratio by the hour of day.

    Args:
    df : pd.DataFrame
        Dataframe of values
    
    Returns:
    None
    '''
    milk_ratio_by_hour = df.groupby("hour_of_day")["milk_ratio"]
    avg_milk_ratio_per_hour = milk_ratio_by_hour.mean()

    plt.figure()
    sns.lineplot(avg_milk_ratio_per_hour, marker="o", color="red")
    plt.title("Average Milk Ratio vs Hour in Day")
    plt.xlabel("Hour in Day")
    plt.ylabel("Average Milk Ratio")
    plt.xticks(range(0,24))
    plt.show()
