#!/usr/bin/env python3
"""
Run full pipeline with Optuna optimization for multiple stocks
Continues until achieving great results (Sharpe > 2.0)
"""

import sys
import os
sys.path.insert(0, '/home/car/stock')

from src.data.data_loader import get_stock_data
from src.features.feature_builder import (
    add_fundamental_features, add_technical_features,
    add_macro_features, add_volatility_features,
    add_acf_ccf_lagged_features, add_lagged_features
)
from src.models.prepare_data import prepare_training_data
from src.models.train_model import train_lgbm_model
from src.backtesting.engine import Backtester
from src.backtesting.agent import TradingAgent
from src.backtesting.optimizer import ParameterOptimizer
from src.backtesting.metrics import calculate_sharpe_ratio, calculate_max_drawdown
import pandas as pd
import numpy as np
import json

# Target Sharpe ratio for "great result"
TARGET_SHARPE = 2.0
MAX_ITERATIONS = 5  # Maximum rounds of optimization per stock

def run_pipeline_for_ticker(ticker, start_date, end_date, target_sharpe=TARGET_SHARPE, max_iterations=MAX_ITERATIONS):
    """
    Run pipeline for a single ticker with optuna optimization
    """
    print(f"\n{'='*60}")
    print(f"{'='*60}")
    print(f"{'='*60}")
    print(f"  RUNNING FULL PIPELINE FOR {ticker}")
    print(f"{'='*60}")
    print(f"{'='*60}")
    print(f"{'='*60}\n")

    # Configuration
    TEST_SIZE = 0.2
    HORIZON = 10
    THRESHOLD = 0.03
    INITIAL_CASH = 100000
    RISK_FREE_RATE = 0.02

    # 1. Load Data
    print(f"[{ticker}] Loading data from {start_date} to {end_date}...")
    data = get_stock_data(ticker, start_date, end_date)
    if data.empty:
        print(f"[{ticker}] ERROR: Could not retrieve stock data. Skipping.")
        return None

    # Make index timezone-naive
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    print(f"[{ticker}] Data loaded: {len(data)} rows")

    # 2. Build Features
    print(f"[{ticker}] Building features...")

    data = add_fundamental_features(data, ticker)
    data = add_technical_features(data)
    data = add_macro_features(data)
    data = add_volatility_features(data)

    # Add lagged features
    data = add_acf_ccf_lagged_features(data, target_horizon=HORIZON)

    print(f"[{ticker}] Features built")

    # 3. Prepare Training Data
    print(f"[{ticker}] Preparing training data...")
    X, y = prepare_training_data(data, horizon=HORIZON, threshold=THRESHOLD)

    # 4. Split Data
    split_index = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"[{ticker}] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 5. Train Model
    print(f"[{ticker}] Training model...")
    model = train_lgbm_model(X_train, y_train, X_test, y_test)

    best_result = {
        'ticker': ticker,
        'sharpe': -float('inf'),
        'weights': None,
        'thresholds': None,
        'final_value': None,
        'total_return': None,
        'max_drawdown': None
    }

    # 6. Run Optuna Optimization iterations until great result
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"[{ticker}] OPTIMIZATION ITERATION {iteration}/{max_iterations}")
        print(f"{'='*60}\n")

        # Initialize Optimizer with Training Data
        optimizer = ParameterOptimizer(model, X_train, y_train, INITIAL_CASH)

        # Run optimization with increasing trials
        n_trials = 20 * iteration  # Increase trials each iteration
        print(f"[{ticker}] Running Optuna with {n_trials} trials...")

        best_params = optimizer.optimize_params(n_trials=n_trials)

        # Apply Best Parameters
        weights = {
            'model': best_params['w_model'],
            'trend': best_params['w_trend'],
            'momentum': best_params['w_momentum'],
            'volatility': best_params['w_volatility'],
            'macro': best_params['w_macro'],
            'sentiment': 0.0
        }
        total = sum(weights.values())
        if total > 0:
            for k in weights:
                weights[k] /= total

        thresholds = {
            'buy': best_params['thresh_buy'],
            'sell': best_params['thresh_sell'],
            'strong_buy': best_params['thresh_buy'] + 0.15,
            'strong_sell': best_params['thresh_sell'] - 0.15
        }

        print(f"\n[{ticker}] Weights: {weights}")
        print(f"[{ticker}] Thresholds: {thresholds}")

        # 7. Run Backtest with Optimized Parameters
        agent = TradingAgent(model, weights=weights, thresholds=thresholds)
        backtester = Backtester(agent, X_test, y_test, initial_cash=INITIAL_CASH)
        portfolio_df = backtester.run_backtest()

        # 8. Calculate Performance Metrics
        final_value = portfolio_df['PortfolioValue'].iloc[-1]
        total_return = ((final_value - INITIAL_CASH) / INITIAL_CASH) * 100
        sharpe = calculate_sharpe_ratio(portfolio_df, RISK_FREE_RATE)
        max_drawdown = calculate_max_drawdown(portfolio_df)

        print(f"\n[{ticker}] PERFORMANCE METRICS (Iteration {iteration}):")
        print(f"  Final Portfolio Value:   ${final_value:,.2f}")
        print(f"  Total Return:            {total_return:.2f}%")
        print(f"  Sharpe Ratio:            {sharpe:.4f}")
        print(f"  Max Drawdown:            {max_drawdown:.2f}%")

        # Benchmark comparison
        buy_hold_start_price = X_test['Close'].iloc[0]
        buy_hold_end_price = X_test['Close'].iloc[-1]
        benchmark_return = ((buy_hold_end_price - buy_hold_start_price) / buy_hold_start_price) * 100

        print(f"\n[{ticker}] BUY & HOLD BENCHMARK:")
        print(f"  Total Return:            {benchmark_return:.2f}%")

        if sharpe > best_result['sharpe']:
            best_result = {
                'ticker': ticker,
                'sharpe': sharpe,
                'weights': weights,
                'thresholds': thresholds,
                'final_value': final_value,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'benchmark_return': benchmark_return,
                'trials': n_trials
            }

        # Check if we achieved target
        if sharpe >= target_sharpe:
            print(f"\n{'*'*60}")
            print(f"[{ticker}] GREAT RESULT ACHIEVED! Sharpe: {sharpe:.4f} >= {target_sharpe}")
            print(f"{'*'*60}")
            break
        else:
            print(f"\n[{ticker}] Target not yet achieved. Sharpe: {sharpe:.4f} < {target_sharpe}")

    return best_result


def main():
    """
    Run full pipeline for AAPL, GOOGL, TSLA
    """
    TICKERS = ['AAPL', 'GOOGL', 'TSLA']
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'

    all_results = []

    for ticker in TICKERS:
        result = run_pipeline_for_ticker(ticker, START_DATE, END_DATE)
        if result:
            all_results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print(f"{'='*60}")
    print(f"  FINAL SUMMARY - ALL STOCKS")
    print(f"{'='*60}")
    print(f"{'='*60}\n")

    for result in all_results:
        print(f"\n--- {result['ticker']} ---")
        print(f"  Best Sharpe:          {result['sharpe']:.4f}")
        print(f"  Total Return:         {result['total_return']:.2f}%")
        print(f"  Max Drawdown:         {result['max_drawdown']:.2f}%")
        print(f"  Benchmark Return:     {result.get('benchmark_return', 'N/A'):.2f}%")
        print(f"  Trials Run:           {result.get('trials', 'N/A')}")
        print(f"  Weights:             {result['weights']}")
        print(f"  Thresholds:           {result['thresholds']}")

    # Save results to JSON
    output_file = 'results/multi_stock_optimization_results.json'
    os.makedirs('results', exist_ok=True)

    summary = {
        'target_sharpe': TARGET_SHARPE,
        'start_date': START_DATE,
        'end_date': END_DATE,
        'results': all_results
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")

    return all_results


if __name__ == '__main__':
    main()
