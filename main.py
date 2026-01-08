from src.data.data_loader import get_stock_data
from src.features.feature_builder import add_fundamental_features, add_technical_features, add_macro_features
from src.models.prepare_data import prepare_training_data
from src.models.train_model import train_lgbm_model
from src.backtesting.engine import Backtester
from src.backtesting.metrics import calculate_sharpe_ratio, calculate_max_drawdown
import pandas as pd

def main():
    """
    Main function to run the entire stock prediction and backtesting pipeline.
    """
    # --- Configuration ---
    TICKER = 'AAPL'
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    TEST_SIZE = 0.2
    HORIZON = 10 # Predict 10 days ahead
    THRESHOLD = 0.03 # 3% price increase for a 'Buy' signal
    INITIAL_CASH = 100000
    RISK_FREE_RATE = 0.02 # Annualized

    print(f"--- Starting Pipeline for {TICKER} ---")

    # 1. Load Data
    data = get_stock_data(TICKER, START_DATE, END_DATE)
    if data.empty:
        print("Could not retrieve stock data. Exiting.")
        return

    # IMPORTANT: Make index timezone-naive for all operations
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # 2. Build Features
    data = add_fundamental_features(data, TICKER)
    data = add_technical_features(data)
    data = add_macro_features(data)
    
    # 3. Prepare Training Data
    X, y = prepare_training_data(data, horizon=HORIZON, threshold=THRESHOLD)

    # 4. Split Data
    split_index = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # 5. Train Model
    model = train_lgbm_model(X, y) # Pass the full dataset for training logic
    
    # 6. Run Backtest
    backtester = Backtester(model, X_test, y_test, initial_cash=INITIAL_CASH)
    portfolio_df = backtester.run_backtest()
    
    # 7. Analyze Performance
    print("\n--- Final Performance Analysis ---")
    
    # Strategy Performance
    final_value_strategy = portfolio_df['PortfolioValue'].iloc[-1]
    total_return_strategy = ((final_value_strategy - INITIAL_CASH) / INITIAL_CASH) * 100
    sharpe_strategy = calculate_sharpe_ratio(portfolio_df, RISK_FREE_RATE)
    max_drawdown_strategy = calculate_max_drawdown(portfolio_df)

    # "Buy and Hold" Benchmark Performance
    buy_hold_start_price = X_test['Close'].iloc[0]
    buy_hold_end_price = X_test['Close'].iloc[-1]
    total_return_benchmark = ((buy_hold_end_price - buy_hold_start_price) / buy_hold_start_price) * 100
    
    benchmark_portfolio = pd.DataFrame({
        'PortfolioValue': (INITIAL_CASH / buy_hold_start_price) * X_test['Close']
    })
    sharpe_benchmark = calculate_sharpe_ratio(benchmark_portfolio, RISK_FREE_RATE)
    max_drawdown_benchmark = calculate_max_drawdown(benchmark_portfolio)

    # Print Report
    print("\n--- Strategy Performance ---")
    print(f"Final Portfolio Value:   ${final_value_strategy:,.2f}")
    print(f"Total Return:            {total_return_strategy:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_strategy:.2f}")
    print(f"Maximum Drawdown:        {max_drawdown_strategy:.2f}%")
    
    print("\n--- Buy & Hold Benchmark ---")
    print(f"Final Portfolio Value:   ${(INITIAL_CASH * (1 + total_return_benchmark/100)):,.2f}")
    print(f"Total Return:            {total_return_benchmark:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_benchmark:.2f}")
    print(f"Maximum Drawdown:        {max_drawdown_benchmark:.2f}%")
    print("\n------------------------------------")


if __name__ == '__main__':
    main()
