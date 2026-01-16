import pandas as pd
import requests
import os
from dotenv import load_dotenv

class EarningsLoader:
    """
    Fetches Earnings Surprise data using Financial Modeling Prep (FMP) API.
    """
    def __init__(self, cache_dir='data/cache/earnings'):
        load_dotenv()
        self.api_key = os.getenv("FMP_API_KEY")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        if not self.api_key:
            print("WARNING: 'FMP_API_KEY' not found in .env. Earnings features will be empty.")
            # Reminder for user: You need a free API key from https://site.financialmodelingprep.com/developer/docs/
    
    def get_earnings_surprises(self, ticker: str, force_update: bool = False) -> pd.DataFrame:
        """
        Fetches historical earnings surprises for a ticker.
        Checks local cache first.
        """
        cache_path = os.path.join(self.cache_dir, f"{ticker}_earnings.csv")
        
        if not force_update and os.path.exists(cache_path):
            print(f"INFO: Loading {ticker} earnings from cache: {cache_path}")
            try:
                df = pd.read_csv(cache_path, index_col='date', parse_dates=True)
                return df
            except Exception as e:
                print(f"WARNING: Corrupt cache file {cache_path}, refetching. Error: {e}")
        
        if not self.api_key:
            return pd.DataFrame()
            
        url = f"https://financialmodelingprep.com/api/v3/earnings-surprises/{ticker}?apikey={self.api_key}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if not data or not isinstance(data, list):
                print(f"WARNING: No earnings data found for {ticker}")
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            
            # Key columns: 'date', 'actualEarningResult', 'estimatedEarning', 'surprise', 'surprisePercent'
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # Save to cache
            df.to_csv(cache_path)
            print(f"Successfully fetched {len(df)} earnings events for {ticker} and cached to {cache_path}.")
            
            return df[['surprise', 'surprisePercent']]
            
        except Exception as e:
            print(f"ERROR: Failed to fetch earnings data: {e}")
            return pd.DataFrame()

if __name__ == '__main__':
    loader = EarningsLoader()
    df = loader.get_earnings_surprises('AAPL')
    print(df.head())
