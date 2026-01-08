import streamlit as st
import pandas as pd
from src.data.data_loader import get_stock_data
from src.features.feature_builder import add_fundamental_features, add_technical_features, add_macro_features
from src.models.prepare_data import prepare_training_data
from src.models.train_model import train_lgbm_model
from src.backtesting.engine import Backtester
from src.backtesting.metrics import calculate_sharpe_ratio, calculate_max_drawdown
import datetime

st.set_page_config(layout="wide")

st.title("Stock Prediction & Backtesting Engine")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Configuration")
    
    ticker = st.text_input("Stock Ticker", "AAPL")
    
    today = datetime.date.today()
    start_date = st.date_input("Start Date", today - datetime.timedelta(days=3*365))
    end_date = st.date_input("End Date", today)
    
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    horizon = st.slider("Prediction Horizon (Days)", 1, 30, 10, 1)
    threshold = st.slider("Buy Signal Threshold (%)", 1.0, 10.0, 3.0, 0.5) / 100.0

    run_button = st.button("Run Backtest")

if run_button:
    with st.status("Running pipeline...", expanded=True) as status:
        # 1. Load Data
        status.update(label="Step 1/5: Loading Data...")
        data = get_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
        if data.empty:
            status.update(label="Failed to load data.", state="error")
            st.error(f"Could not retrieve stock data for {ticker}. Please check the ticker symbol.")
        else:
            # 2. Build Features
            status.update(label="Step 2/5: Building Features...")
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            data = add_fundamental_features(data, ticker)
            data = add_technical_features(data)
            data = add_macro_features(data)
            
            # 3. Prepare Training Data
            status.update(label="Step 3/5: Preparing Training Data...")
            X, y = prepare_training_data(data, horizon=horizon, threshold=threshold)

            # 4. Train Model
            status.update(label="Step 4/5: Training Model...")
            split_index = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            model = train_lgbm_model(X, y)
            
            # 5. Run Backtest
            status.update(label="Step 5/5: Running Backtest...")
            backtester = Backtester(model, X_test, y_test)
            portfolio_df = backtester.run_backtest()
            
            status.update(label="Pipeline complete!", state="complete")

    st.success("Backtest finished. See results below.")

    # --- Display Results ---
    st.header("Performance Analysis")

    # Strategy Performance
    final_value_strategy = portfolio_df['PortfolioValue'].iloc[-1]
    total_return_strategy = ((final_value_strategy - 100000) / 100000) * 100
    sharpe_strategy = calculate_sharpe_ratio(portfolio_df)
    max_drawdown_strategy = calculate_max_drawdown(portfolio_df)

    # Benchmark Performance
    buy_hold_start_price = X_test['Close'].iloc[0]
    buy_hold_end_price = X_test['Close'].iloc[-1]
    total_return_benchmark = ((buy_hold_end_price - buy_hold_start_price) / buy_hold_start_price) * 100
    benchmark_portfolio = pd.DataFrame({'PortfolioValue': (100000 / buy_hold_start_price) * X_test['Close']})
    sharpe_benchmark = calculate_sharpe_ratio(benchmark_portfolio)
    max_drawdown_benchmark = calculate_max_drawdown(benchmark_portfolio)

    # Display Metrics in Columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Our Strategy")
        st.metric("Total Return", f"{total_return_strategy:.2f}%")
        st.metric("Sharpe Ratio", f"{sharpe_strategy:.2f}")
        st.metric("Max Drawdown", f"{max_drawdown_strategy:.2f}%")
        st.metric("Final Value", f"${final_value_strategy:,.2f}")

    with col2:
        st.subheader("Buy & Hold Benchmark")
        st.metric("Total Return", f"{total_return_benchmark:.2f}%")
        st.metric("Sharpe Ratio", f"{sharpe_benchmark:.2f}")
        st.metric("Max Drawdown", f"{max_drawdown_benchmark:.2f}%")
        st.metric("Final Value", f"${(100000 * (1 + total_return_benchmark/100)):,.2f}")

    # Display Chart
    st.header("Portfolio Value Over Time")
    chart_df = pd.DataFrame({
        'Strategy': portfolio_df['PortfolioValue'],
        'Buy & Hold': benchmark_portfolio['PortfolioValue']
    })
    st.line_chart(chart_df)

    # Display DataFrames
    with st.expander("Show Final Data with Features"):
        st.dataframe(data.tail())

else:
    st.info("Configure parameters in the sidebar and click 'Run Backtest' to start.")
