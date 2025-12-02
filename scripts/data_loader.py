"""
Data loading and preprocessing utilities for coffee sales analysis.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load coffee sales data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with coffee sales data
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_datetime(df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
    """
    Convert datetime column to pandas datetime type if needed.
    
    Args:
        df: Input DataFrame
        datetime_col: Name of the datetime column
        
    Returns:
        DataFrame with converted datetime column
    """
    df = df.copy()
    if not np.issubdtype(df[datetime_col].dtype, np.datetime64):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    return df


def normalize_coffee_names(df: pd.DataFrame, coffee_col: str = 'coffee_name') -> pd.DataFrame:
    """
    Normalize coffee names by merging similar variants.
    
    Args:
        df: Input DataFrame
        coffee_col: Name of the coffee name column
        
    Returns:
        DataFrame with normalized coffee names in 'new_coffee_name' column
    """
    df = df.copy()
    df['new_coffee_name'] = df[coffee_col].str.replace('Americano with Milk', 'Americano')
    df['new_coffee_name'] = df['new_coffee_name'].str.replace('Cocoa', 'Hot Chocolate')
    return df


def prepare_daily_sales(df: pd.DataFrame, datetime_col: str = 'datetime', 
                       sales_col: str = 'money') -> pd.Series:
    """
    Aggregate sales by day.
    
    Args:
        df: Input DataFrame
        datetime_col: Name of the datetime column
        sales_col: Name of the sales amount column
        
    Returns:
        Series with daily sales totals
    """
    df = preprocess_datetime(df, datetime_col)
    daily_sales = df.groupby(pd.Grouper(key=datetime_col, freq='D'))[sales_col].sum()
    return daily_sales


def prepare_daily_coffee_sales(df: pd.DataFrame, date_col: str = 'date',
                               coffee_col: str = 'new_coffee_name') -> pd.DataFrame:
    """
    Create a pivot table of daily sales by coffee type.
    
    Args:
        df: Input DataFrame
        date_col: Name of the date column
        coffee_col: Name of the coffee name column
        
    Returns:
        DataFrame with dates as index and coffee types as columns
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    daily_coffee_sales = df.groupby([date_col, coffee_col]).size().unstack(fill_value=0)
    return daily_coffee_sales

