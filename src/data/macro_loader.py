import pandas_datareader.data as web
import pandas as pd
from datetime import datetime

def get_interest_rate_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches the Federal Funds Effective Rate from FRED and processes it.

    Args:
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame with a daily time series of the 
                      forward-filled interest rate.
    """
    try:
        # Fetch the daily effective federal funds rate
        dff = web.DataReader('DFF', 'fred', start=start_date, end=end_date)
        
        # Create a full date range to handle weekends and holidays
        daily_index = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Reindex the data to the daily frequency and forward-fill missing values
        dff_daily = dff.reindex(daily_index).ffill()
        
        dff_daily.index.name = 'Date'
        dff_daily = dff_daily.rename(columns={'DFF': 'Interest_Rate'})
        
        print(f"Successfully fetched and processed interest rate data from {start_date} to {end_date}.")
        return dff_daily

    except Exception as e:
        print(f"An error occurred while fetching interest rate data: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example usage:
    start = '2023-01-01'
    end = '2024-01-01'
    
    interest_rate_data = get_interest_rate_data(start, end)
    
    if not interest_rate_data.empty:
        print("\nFirst 5 rows of interest rate data:")
        print(interest_rate_data.head())
        print("\nLast 5 rows of interest rate data:")
        print(interest_rate_data.tail())
        
        # Check a date that would have been a weekend to see the ffill
        print("\nChecking a weekend date (2023-01-07):")
        print(interest_rate_data.loc['2023-01-07'])
