import pandas as pd
from src.features.feature_builder import add_fundamental_features, add_technical_features, add_macro_features
from src.data.data_loader import get_stock_data

def prepare_training_data(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.02) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepares the final training data by creating the target variable.

    Args:
        df (pd.DataFrame): DataFrame with all features.
        horizon (int): The number of days to look into the future for the target.
        threshold (float): The percentage change required to classify as a 'Buy'.

    Returns:
        (pd.DataFrame, pd.Series): A tuple of (features, target).
    """
    # Calculate the future price
    df['Future_Close'] = df['Close'].shift(-horizon)
    
    # Calculate the percentage change
    df['Pct_Change'] = (df['Future_Close'] - df['Close']) / df['Close']
    
    # Create the target variable
    df['Target'] = (df['Pct_Change'] > threshold).astype(int)
    
    # Drop rows with NaNs created by the shift operation and intermediate columns
    df.dropna(subset=['Future_Close'], inplace=True)
    df.drop(columns=['Future_Close', 'Pct_Change'], inplace=True)
    
    # Separate features and target
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    print("OK: Training data prepared.")
    print(f"Feature set shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    
    return X, y

if __name__ == '__main__':
    TICKER = 'MSFT'
    START = '2022-01-01' # Using a longer period to ensure enough data for target creation
    END = '2024-01-01'
    
    data = get_stock_data(TICKER, START, END)
    
    if not data.empty:
        # Build features
        data = add_fundamental_features(data, TICKER)
        data = add_technical_features(data)
        data = add_macro_features(data)
        
        # Prepare training data
        X, y = prepare_training_data(data, horizon=5, threshold=0.02)
        
        print("\n--- Features (X) ---")
        print(X.tail())
        
        print("\n--- Target (y) ---")
        print(y.tail())
