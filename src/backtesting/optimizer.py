import optuna
import pandas as pd
import numpy as np
from src.backtesting.engine import Backtester
from src.backtesting.agent import TradingAgent
from src.backtesting.metrics import calculate_sharpe_ratio

class ParameterOptimizer:
    """
    Optimizes TradingAgent parameters using Optuna.
    Supports standard cross-validation and Walk-Forward Optimization (WFO).
    """
    def __init__(self, model, X, y, initial_cash=100000):
        self.model = model
        self.X = X
        self.y = y
        self.initial_cash = initial_cash

    def objective(self, trial):
        """
        Optuna objective function for a single optimization run.
        Optimizes for Sharpe Ratio.
        """
        # 1. Suggest Parameters
        weights = {
            'model': trial.suggest_float('w_model', 0.3, 0.8),
            'trend': trial.suggest_float('w_trend', 0.0, 0.4),
            'momentum': trial.suggest_float('w_momentum', 0.0, 0.4),
            'volatility': trial.suggest_float('w_volatility', 0.0, 0.4),
            'macro': trial.suggest_float('w_macro', 0.0, 0.3),
            'sentiment': 0.0 # Fixed for now
        }
        
        # Normalize weights to sum to 1 (optional, but good practice)
        total_weight = sum(weights.values())
        if total_weight > 0:
            for k in weights:
                weights[k] /= total_weight
        
        thresholds = {
            'buy': trial.suggest_float('thresh_buy', 0.55, 0.85),
            'sell': trial.suggest_float('thresh_sell', 0.15, 0.45),
            'strong_buy': 0.0, # Will set dynamically based on buy
            'strong_sell': 0.0 # Will set dynamically based on sell
        }
        
        # Enforce logical constraints
        thresholds['strong_buy'] = thresholds['buy'] + 0.15
        thresholds['strong_sell'] = thresholds['sell'] - 0.15
        
        # 2. Run Backtest
        # We use a simplified in-sample split here, or the full provided dataset
        # In WFO, X and y passed to this method are the TRAIN set of the current fold.
        
        agent = TradingAgent(self.model, weights=weights, thresholds=thresholds)
        backtester = Backtester(agent, self.X, self.y, initial_cash=self.initial_cash)
        
        # Suppress print output during optimization
        import sys
        import os
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            portfolio_df = backtester.run_backtest()
        finally:
            sys.stdout = old_stdout
            
        # 3. Calculate Metric (Sharpe Ratio)
        sharpe = calculate_sharpe_ratio(portfolio_df)
        
        # Add penalty for too few trades (overfitting to noise)
        if len(backtester.trade_log) < 5:
            return -1.0
            
        return sharpe

    def optimize_params(self, n_trials=20):
        """
        Runs the optimization process.
        """
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        print("\n--- Optimization Results ---")
        print(f"Best Sharpe Ratio: {study.best_value:.4f}")
        print("Best Parameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v:.4f}")
            
        return study.best_params

    def run_walk_forward_optimization(self, n_splits=5, train_size=0.6, n_trials=10):
        """
        Performs Walk-Forward Optimization.
        
        Logic:
        1. Split data into time-series folds.
        2. For each fold:
           - Train/Optimize params on 'In-Sample' (IS) data.
           - Test best params on 'Out-of-Sample' (OOS) data.
        3. Stitch OOS results to get realistic performance.
        """
        print("\n--- Starting Walk-Forward Optimization ---")
        
        # Simple time-series split
        total_len = len(self.X)
        fold_size = total_len // (n_splits + 1)
        
        oos_results = []
        
        for i in range(n_splits):
            # Define Windows
            # Train: Start to (i+1)*fold_size
            # Test:  (i+1)*fold_size to (i+2)*fold_size
            
            train_end = (i + 1) * fold_size
            test_end = (i + 2) * fold_size
            
            if test_end > total_len:
                test_end = total_len
            
            X_train_fold = self.X.iloc[:train_end]
            y_train_fold = self.y.iloc[:train_end]
            X_test_fold = self.X.iloc[train_end:test_end]
            y_test_fold = self.y.iloc[train_end:test_end]
            
            print(f"\nFold {i+1}/{n_splits}: Train [0:{train_end}] -> Test [{train_end}:{test_end}]")
            
            # Optimize on In-Sample (Train)
            optimizer_fold = ParameterOptimizer(self.model, X_train_fold, y_train_fold, self.initial_cash)
            best_params = optimizer_fold.optimize_params(n_trials=n_trials)
            
            # Construct Best Agent
            # (Need to reconstruct weights/thresholds dict from flat params)
            weights = {
                'model': best_params['w_model'],
                'trend': best_params['w_trend'],
                'momentum': best_params['w_momentum'],
                'volatility': best_params['w_volatility'],
                'macro': best_params['w_macro'],
                'sentiment': 0.0
            }
            # Normalize
            total = sum(weights.values())
            if total > 0:
                for k in weights: weights[k] /= total
                
            thresholds = {
                'buy': best_params['thresh_buy'],
                'sell': best_params['thresh_sell'],
                'strong_buy': best_params['thresh_buy'] + 0.15,
                'strong_sell': best_params['thresh_sell'] - 0.15
            }
            
            # Evaluate on Out-of-Sample (Test)
            agent = TradingAgent(self.model, weights=weights, thresholds=thresholds)
            
            # We need to carry over cash from previous fold to make equity curve continuous?
            # Simpler: Run backtest with initial cash, then calculate returns and stitch returns.
            # Or just sum up the trade logs.
            
            backtester = Backtester(agent, X_test_fold, y_test_fold, initial_cash=self.initial_cash)
            fold_portfolio = backtester.run_backtest()
            
            # Store OOS performance
            oos_results.append(fold_portfolio)
            
        # Stitch Results
        if oos_results:
            full_oos_df = pd.concat(oos_results)
            print("\n--- Walk-Forward Optimization Complete ---")
            return full_oos_df
        else:
            return pd.DataFrame()
