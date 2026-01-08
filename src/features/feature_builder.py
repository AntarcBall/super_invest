import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
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

def add_lagged_features(df: pd.DataFrame, target_col: str = 'Close', max_lag: int = 10) -> pd.DataFrame:
    """
    Analyzes cross-correlation to find predictive lags and adds them as features.
    
    Args:
        df: DataFrame with features.
        target_col: The column to predict (future return based on Close).
        max_lag: Maximum number of days to look back.
    """
    print("INFO: Starting time-lagged feature engineering...")
    try:
        # Calculate a proxy target: Future 5-day return
        # We use this to find which past features correlate with future price movement
        future_return = df[target_col].shift(-5).pct_change(5)
        
        # List of features to check for lags
        features_to_lag = ['RSI_14', 'MACD_12_26_9', 'ATR_14', 'VIX', 'Volume']
        
        for feature in features_to_lag:
            if feature not in df.columns:
                continue
                
            best_lag = 0
            best_corr = 0
            
            # Check correlations for lags 1 to max_lag
            for lag in range(1, max_lag + 1):
                # Correlation between Feature(t-lag) and FutureReturn(t)
                # We align them by shifting the feature
                lagged_feature = df[feature].shift(lag)
                # Ensure numeric types for correlation
                corr = pd.to_numeric(lagged_feature, errors='coerce').corr(pd.to_numeric(future_return, errors='coerce'))
                
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            
            # If we found a meaningful correlation (> 0.05), add the feature
            if abs(best_corr) > 0.05:
                col_name = f"{feature}_lag_{best_lag}"
                df[col_name] = df[feature].shift(best_lag)
                print(f"   -> Added {col_name} (Corr: {best_corr:.3f})")
                
        df.fillna(0, inplace=True)
        print("OK: Lagged features added.")
        return df

    except Exception as e:
        print(f"ERROR: Could not add lagged features: {e}")
        return df

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds volatility features (ATR and VIX) to the stock data DataFrame."""
    print("INFO: Starting volatility feature engineering...")
    try:
        # 1. Calculate ATR (Average True Range) using pandas-ta
        # Default period is usually 14
        df.ta.atr(length=14, append=True)
        
        # 2. Fetch VIX (Market Volatility Index)
        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
        
        # We need to handle potential download errors gracefully
        try:
            vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not vix_data.empty:
                # Flatten MultiIndex columns if present
                if isinstance(vix_data.columns, pd.MultiIndex):
                    vix_data.columns = vix_data.columns.get_level_values(0)
                
                # Keep only the Close column and rename it
                # Check if 'Close' exists, if not try to use the first column
                if 'Close' in vix_data.columns:
                    vix_data = vix_data[['Close']].rename(columns={'Close': 'VIX'})
                else:
                     vix_data = vix_data.iloc[:, [0]].rename(columns={vix_data.columns[0]: 'VIX'})

                
                # Ensure timezone-naive index for merging
                if vix_data.index.tz is not None:
                    vix_data.index = vix_data.index.tz_localize(None)
                
                # Merge VIX data
                df = df.merge(vix_data, how='left', left_index=True, right_index=True)
                df['VIX'] = df['VIX'].ffill().fillna(0) # Forward fill missing VIX days
            else:
                print("WARNING: VIX data is empty. Filling with 0.")
                df['VIX'] = 0.0
        except Exception as e:
            print(f"WARNING: Could not fetch VIX data: {e}. Filling with 0.")
            df['VIX'] = 0.0

        # Fill any NaNs created by ATR (first 14 days)
        df.fillna(0, inplace=True)
        
        print("OK: Volatility features (ATR, VIX) added.")
        return df

    except Exception as e:
        print(f"ERROR: Could not add volatility features: {e}")
        return df

def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds macroeconomic features (interest rate) to the stock data DataFrame."""
    print("INFO: Starting macroeconomic analysis...")
    try:
        # Ensure the index is a DatetimeIndex before proceeding
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")

        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
        rates = get_interest_rate_data(start_date, end_date)
        
        # Ensure the original dataframe's index is timezone-naive for the merge
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        df = df.merge(rates, how='left', left_index=True, right_index=True)
        
        # Use assignment instead of chained inplace calls
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
        data = add_volatility_features(data)
        data = add_macro_features(data)
        data = add_lagged_features(data)
        
        print("\n--- Final DataFrame with all features (last 5 rows) ---")
        print(data.tail())
        print("\nColumns:")
        print(data.columns)
        print("\nData Types:")
        print(data.info())

