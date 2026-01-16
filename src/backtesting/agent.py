import numpy as np
import pandas as pd

class TradingAgent:
    """
    An 'Intelligent' Trading Agent that makes decisions based on multiple factors:
    1. ML Model Probability (Confidence)
    2. Technical Analysis (RSI, Bollinger Bands, Trend)
    3. Volatility (VIX, ATR)
    4. Macroeconomic context (Interest Rates)
    
    It combines these into a single 'Conviction Score' and determines:
    - Action: BUY, SELL, HOLD
    - Sizing: Fraction of portfolio to deploy
    """
    
    def __init__(self, model, weights=None, thresholds=None):
        """
        Args:
            model: Trained ML model (must have predict_proba)
            weights: Dict of weights for scoring components
            thresholds: Dict of thresholds for decisions
        """
        self.model = model
        
        # Default Weights
        self.weights = weights or {
            'model': 0.5,       # ML Model confidence is primary
            'trend': 0.2,       # SMA Trend alignment
            'momentum': 0.15,   # RSI
            'volatility': 0.1,  # VIX / ATR (Low vol is good for stability?)
            'macro': 0.05       # Interest rate regime
        }
        
        # Default Thresholds
        self.thresholds = thresholds or {
            'strong_buy': 0.75,
            'buy': 0.60,
            'sell': 0.40,
            'strong_sell': 0.25
        }

    def get_signal_score(self, market_data):
        """
        Calculates a composite conviction score between 0 and 1.
        """
        # 1. Model Confidence (0 to 1)
        # Handle both sklearn (predict_proba) and direct output
        try:
            # Expecting row to be shaped (1, n_features)
            X = market_data.to_frame().T
            if hasattr(self.model, 'predict_proba'):
                prob = self.model.predict_proba(X)[0][1]
            else:
                prob = float(self.model.predict(X)[0]) # Fallback for binary or regression
        except Exception:
            prob = 0.5 # Neutral if model fails
            
        # 2. Trend Score (SMA alignment)
        trend_score = 0.5
        if 'SMA_50' in market_data and 'SMA_200' in market_data:
            close = market_data.get('Close', 0)
            sma50 = market_data['SMA_50']
            sma200 = market_data['SMA_200']
            
            if close > sma50 > sma200:
                trend_score = 1.0 # Strong Uptrend
            elif close < sma50 < sma200:
                trend_score = 0.0 # Strong Downtrend
            elif close > sma200:
                trend_score = 0.7 # Moderate Uptrend
            else:
                trend_score = 0.3
                
        # 3. Momentum Score (RSI)
        # RSI < 30 is oversold (Buy?), RSI > 70 is overbought (Sell?)
        # But for trend following, RSI > 50 is bullish.
        # Let's use a mean-reversion logic tailored for entry:
        # Buy dips in uptrend.
        momentum_score = 0.5
        if 'RSI_14' in market_data:
            rsi = market_data['RSI_14']
            # Map RSI 30-70 to score. 
            # If we are trend following, we want RSI to be healthy (40-60).
            # If we are contrarian, we buy at 30.
            # Let's simple normalize: RSI 50 = 0.5 score. Higher RSI = Higher Score (Bullish momentum)
            # But cap extreme overbought.
            if rsi > 80:
                momentum_score = 0.2 # Overbought, risk of reversal
            elif rsi < 20:
                momentum_score = 0.8 # Oversold, buy opportunity
            else:
                momentum_score = rsi / 100.0

        # 4. Volatility Score (VIX)
        # Lower VIX is generally bullish for equities
        vol_score = 0.5
        if 'VIX' in market_data:
            vix = market_data['VIX']
            # VIX usually 10-30. 
            # VIX < 15 -> Bullish (Score 0.8)
            # VIX > 30 -> Bearish (Score 0.2)
            vol_score = 1.0 - (min(max(vix, 10), 50) - 10) / 40.0

        # Composite Score Calculation
        final_score = (
            prob * self.weights['model'] +
            trend_score * self.weights['trend'] +
            momentum_score * self.weights['momentum'] +
            vol_score * self.weights['volatility']
        )
        
        return final_score

    def decide(self, market_data, portfolio_cash, portfolio_value):
        """
        Decides on action and sizing.
        Returns:
            action (str): 'BUY', 'SELL', 'HOLD'
            size (float): Fraction of portfolio value to trade (0.0 to 1.0)
        """
        score = self.get_signal_score(market_data)
        
        # Determine Action
        action = 'HOLD'
        if score >= self.thresholds['buy']:
            action = 'BUY'
        elif score <= self.thresholds['sell']:
            action = 'SELL'
            
        # Determine Sizing (Dynamic)
        # Base size on conviction (Score)
        # If score is 0.9 (Strong Buy), size should be larger than 0.6 (Weak Buy)
        
        size = 0.0
        if action == 'BUY':
            # Linear scaling from threshold to 1.0
            # score 0.6 -> size 0.2
            # score 1.0 -> size 1.0
            excess_conviction = (score - self.thresholds['buy']) / (1.0 - self.thresholds['buy'])
            size = 0.2 + (0.8 * excess_conviction) # Min 20%, Max 100% of available cash?
            
            # Volatility Adjustment (Kelly-like)
            # Reduce size if ATR is high relative to price
            if 'ATR_14' in market_data and 'Close' in market_data:
                atr_pct = market_data['ATR_14'] / market_data['Close']
                if atr_pct > 0.03: # High volatility (>3% daily move)
                    size *= 0.5 # Cut position size in half
                    
        elif action == 'SELL':
            # For selling, score 0.4 -> size 0.2
            # score 0.0 -> size 1.0
            excess_conviction = (self.thresholds['sell'] - score) / self.thresholds['sell']
            size = 0.5 + (0.5 * excess_conviction) # Min 50%, Max 100% of position
        
        return action, size, score
