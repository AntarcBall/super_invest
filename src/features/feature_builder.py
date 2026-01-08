import pandas as pd
import yfinance as yf
import pandas_ta as ta
from src.data.data_loader import get_stock_data
from src.data.macro_loader import get_interest_rate_data

def add_fundamental_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    try:
        ticker_info = yf.Ticker(ticker).info
        features = {
            'MarketCap': ticker_info.get('marketCap'), 'TrailingPE': ticker_info.get('trailingPE'),
            'ForwardPE': ticker_info.get('forwardPE'), 'PriceToBook': ticker_info.get('priceToBook'),
            'DividendYield': ticker_info.get('dividendYield')
        }
        for name, value in features.items():
            df[name] = value if value is not None else 0
        print(f"OK: Fundamental features added for {ticker}.")
        return df
    except Exception as e:
        print(f"ERROR: Could not add fundamental features for {ticker}: {e}")
        return df

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.ta.rsi(append=True)
        df.ta.macd(append=True)
        df.ta.bbands(append=True)
        df.ta.sma(length=50, append=True)
        df.ta.sma(length=200, append=True)
        df.fillna(0, inplace=True)
        print("OK: Technical features added.")
        return df
    except Exception as e:
        print(f"ERROR: Could not add technical features: {e}")
        return df

def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    print("INFO: Starting macroeconomic analysis...")
    try:
        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
        rates = get_interest_rate_data(start_date, end_date)
        
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        df = df.merge(rates, how='left', left_index=True, right_index=True)
        df['Interest_Rate'] = df['Interest_Rate'].ffill()
        df['Interest_Rate'] = df['Interest_Rate'].fillna(0)
        print("OK: Macroeconomic features added.")
        return df
    except Exception as e:
        print(f"ERROR: Could not add macroeconomic features: {e}")
        df['Interest_Rate'] = 0.0
        return df

if __name__ == '__main__':
    TICKER = 'MSFT'
    START_DATE = '2023-01-01'
    END_DATE = '2024-01-01'
    
    data = get_stock_data(TICKER, START_DATE, END_DATE)
    
    if not data.empty:
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        data = add_fundamental_features(data, TICKER)
        data = add_technical_features(data)
        data = add_macro_features(data)
        
        print("\n--- Final DataFrame with all features (last 5 rows) ---")
        print(data.tail())
        print("\nColumns:")
        print(data.columns)
        print("\nData Types:")
        print(data.info())

