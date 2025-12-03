import pandas as pd

def determine_milk_ratio(drink):
    '''
    Helper function to define milk ratios and return corresponding milk ratios.

    Params:
    drink : str
        Type of drink
    
    Returns:
    tuple
        Tuple corresponding to milk to coffee ratio
    '''
    milk_map = {
        "Latte": (0.7, 0.3),
        "Cappuccino": (0.67, 0.33),
        "Flat White": (0.75, 0.25),
        "Hot Chocolate": (1.0, 0.0),
        "Cocoa": (1.0, 0.0),
        "Americano": (0.0, 1.0),
        "Americano with Milk": (0.2, 0.8),
        "Cortado": (0.5, 0.5)
    }

    milk, coffee = milk_map.get(drink, (0.0, 1.0))
    return milk / (milk + coffee)

def add_hour_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 'hour_of_day' column extracted from the 'datetime' column.

    Params:
        df (pd.DataFrame): 
            Input DataFrame

    Returns:
        pd.DataFrame:
            new 'hour_of_day' column
    """

    df["datetime"] = pd.to_datetime(df["datetime"])

    # Extract hour
    df["hour_of_day"] = df["datetime"].dt.hour


