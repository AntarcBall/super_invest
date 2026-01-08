import yfinance as yf
import pandas as pd

def get_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing the OHLCV data,
                      or an empty DataFrame if the ticker is invalid or data is not found.
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            print(f"Warning: No data found for ticker '{ticker}' for the given date range.")
            return pd.DataFrame()
        
        # Ensure standard column names
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        
        print(f"Successfully fetched data for {ticker} from {start_date} to {end_date}.")
        return data

    except Exception as e:
        print(f"An error occurred while fetching data for ticker '{ticker}': {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example usage:
    ticker = 'AAPL'
    start = '2023-01-01'
    end = '2024-01-01'
    
    aapl_data = get_stock_data(ticker, start, end)
    
    if not aapl_data.empty:
        print("\nFirst 5 rows of AAPL data:")
        print(aapl_data.head())
        print("\nLast 5 rows of AAPL data:")
        print(aapl_data.tail())
        print(f"\nData shape: {aapl_data.shape}")
