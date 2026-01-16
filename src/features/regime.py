import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import joblib

class RegimeDetector:
    """
    Detects market regimes (Bull, Bear, Sideways) using Hidden Markov Models (HMM).
    """
    def __init__(self, n_states=3, covariance_type="full", n_iter=1000, random_state=42):
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states, 
            covariance_type=covariance_type, 
            n_iter=n_iter, 
            random_state=random_state
        )
        self.regime_mapping = {}

    def prepare_features(self, data, window=20):
        """
        Prepares features for HMM: Returns and Volatility.
        """
        df = data.copy()
        # Log Returns
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility (Rolling Standard Deviation of Returns)
        df['volatility'] = df['log_ret'].rolling(window=window).std()
        
        # Drop NaNs
        df.dropna(subset=['log_ret', 'volatility'], inplace=True)
        
        # Features for HMM: [Returns, Volatility]
        X = df[['log_ret', 'volatility']].values
        return X, df.index

    def fit(self, data):
        """
        Fits the HMM model to the data.
        """
        X, _ = self.prepare_features(data)
        self.model.fit(X)
        
        # Analyze states to map them to human-readable regimes
        # We predict the state for each observation
        hidden_states = self.model.predict(X)
        
        # Calculate mean return and volatility for each state to identify them
        state_stats = []
        for i in range(self.n_states):
            mask = (hidden_states == i)
            mean_ret = X[mask, 0].mean()
            mean_vol = X[mask, 1].mean()
            state_stats.append({'state': i, 'mean_ret': mean_ret, 'mean_vol': mean_vol})
            
        stats_df = pd.DataFrame(state_stats)
        
        # Heuristic Mapping:
        # - Highest Return -> Bull
        # - Lowest Return (negative) -> Bear
        # - Middle / Low Volatility -> Sideways
        
        # Sort by Mean Return
        sorted_by_ret = stats_df.sort_values('mean_ret', ascending=False)
        bull_state = sorted_by_ret.iloc[0]['state']
        bear_state = sorted_by_ret.iloc[-1]['state']
        
        # Remaining state is Sideways
        remaining = set(range(self.n_states)) - {bull_state, bear_state}
        sideways_state = list(remaining)[0] if remaining else -1
        
        self.regime_mapping = {
            bull_state: 'Bull',
            bear_state: 'Bear',
            sideways_state: 'Sideways'
        }
        
        print(f"Regime Mapping Identified: {self.regime_mapping}")
        print("State Stats:\n", stats_df)
        
        return self

    def predict_regime(self, data):
        """
        Predicts the regime for the provided data.
        Returns a Series of regimes aligned with the data index.
        """
        X, indices = self.prepare_features(data)
        hidden_states = self.model.predict(X)
        
        regimes = [self.regime_mapping.get(s, 'Unknown') for s in hidden_states]
        return pd.Series(regimes, index=indices, name='Regime')

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
