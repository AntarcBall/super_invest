import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data.data_loader import get_stock_data
from src.features.feature_builder import add_fundamental_features, add_technical_features, add_macro_features, add_volatility_features, add_lagged_features, add_acf_ccf_lagged_features
from src.features.acf_ccf_analysis import analyze_acf, analyze_ccf, multi_feature_ccf_analysis, suggest_lag_features, plot_acf, plot_ccf, get_summary_statistics
from src.models.prepare_data import prepare_training_data
from src.models.train_model import train_lgbm_model
from src.backtesting.engine import Backtester
from src.backtesting.metrics import calculate_sharpe_ratio, calculate_max_drawdown
import datetime

st.set_page_config(layout="wide")

st.title("Stock Prediction & Backtesting Engine")

tab1, tab2, tab3 = st.tabs(["Backtest", "Feature Analysis", "Model Insights"])

with st.sidebar:
    st.header("Configuration")

    ticker = st.text_input("Stock Ticker", "AAPL")

    today = datetime.date.today()
    start_date = st.date_input("Start Date", today - datetime.timedelta(days=3*365))
    end_date = st.date_input("End Date", today)

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    horizon = st.slider("Prediction Horizon (Days)", 1, 30, 10, 1)
    threshold = st.slider("Buy Signal Threshold (%)", 1.0, 10.0, 3.0, 0.5) / 100.0
    use_acf_ccf_features = st.checkbox("Use ACF/CCF-Based Lag Features", value=True, help="Enhanced feature engineering using statistical analysis")

    run_button = st.button("Run Backtest")

with tab1:
    st.header("Backtesting Engine")
    st.info("Configure parameters in the sidebar and click 'Run Backtest' to start.")

    if run_button:
        with st.status("Running pipeline...", expanded=True) as status:
            status.update(label="Step 1/6: Loading Data...")
            data = get_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

            if data.empty:
                status.update(label="Failed to load data.", state="error")
                st.error(f"Could not retrieve stock data for {ticker}. Please check the ticker symbol.")
            else:
                status.update(label="Step 2/6: Building Features...")
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                data = add_fundamental_features(data, ticker)
                data = add_technical_features(data)
                data = add_macro_features(data)
                data = add_volatility_features(data)

                status.update(label="Step 3/6: Adding Lagged Features...")
                if use_acf_ccf_features:
                    data = add_acf_ccf_lagged_features(data, target_horizon=horizon)
                else:
                    data = add_lagged_features(data)

                status.update(label="Step 4/6: Preparing Training Data...")
                X, y = prepare_training_data(data, horizon=horizon, threshold=threshold)

                status.update(label="Step 5/6: Training Model...")
                split_index = int(len(X) * (1 - test_size))
                X_train, X_test = X[:split_index], X[split_index:]
                y_train, y_test = y[:split_index], y[split_index:]

                model = train_lgbm_model(X_train, y_train, X_test, y_test)

                status.update(label="Step 6/6: Running Backtest...")
                backtester = Backtester(model, X_test, y_test)
                portfolio_df = backtester.run_backtest()

                status.update(label="Pipeline complete!", state="complete")

                st.success("Backtest finished. See results below.")

                st.session_state['data'] = data
                st.session_state['model'] = model
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test

                st.header("Performance Analysis")

                final_value_strategy = portfolio_df['PortfolioValue'].iloc[-1]
                total_return_strategy = ((final_value_strategy - 100000) / 100000) * 100
                sharpe_strategy = calculate_sharpe_ratio(portfolio_df)
                max_drawdown_strategy = calculate_max_drawdown(portfolio_df)

                buy_hold_start_price = X_test['Close'].iloc[0]
                buy_hold_end_price = X_test['Close'].iloc[-1]
                total_return_benchmark = ((buy_hold_end_price - buy_hold_start_price) / buy_hold_start_price) * 100
                benchmark_portfolio = pd.DataFrame({'PortfolioValue': (100000 / buy_hold_start_price) * X_test['Close']})
                sharpe_benchmark = calculate_sharpe_ratio(benchmark_portfolio)
                max_drawdown_benchmark = calculate_max_drawdown(benchmark_portfolio)

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

                st.header("Portfolio Value Over Time")
                chart_df = pd.DataFrame({
                    'Strategy': portfolio_df['PortfolioValue'],
                    'Buy & Hold': benchmark_portfolio['PortfolioValue']
                })
                st.line_chart(chart_df)

                with st.expander("Show Final Data with Features"):
                    st.dataframe(data.tail())

with tab2:
    st.header("Feature Analysis (ACF/CCF)")

    if 'data' in st.session_state:
        data = st.session_state['data']

        st.subheader("Autocorrelation Analysis (ACF)")
        st.info("ACF helps identify temporal patterns and dependencies in stock prices.")

        acf_nlags = st.slider("ACF Lags to Analyze", 10, 100, 40, 5)

        try:
            autocorr, confint, acf_insights = analyze_acf(data, 'Close', nlags=acf_nlags)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Trend Strength (Lag 1)", f"{acf_insights['trend_strength']:.3f}")
            with col2:
                st.metric("Strongest Lag", f"{acf_insights['strongest_lag']} days")

            st.text(get_summary_statistics('ACF', acf_insights))

            acf_fig = plot_acf(autocorr, confint, title=f"Autocorrelation of {ticker} Closing Prices")
            st.pyplot(acf_fig)
            plt.close(acf_fig)
        except Exception as e:
            st.error(f"Error in ACF analysis: {e}")

        st.divider()

        st.subheader("Cross-Correlation Analysis (CCF)")
        st.info("CCF identifies which features are most predictive of future returns at different time lags.")

        features_to_analyze = ['RSI_14', 'MACD_12_26_9', 'ATR_14', 'VIX', 'Volume']
        available_features = [f for f in features_to_analyze if f in data.columns]

        if available_features:
            selected_features = st.multiselect(
                "Select Features to Analyze",
                available_features,
                default=['RSI_14', 'MACD_12_26_9']
            )

            ccf_nlags = st.slider("CCF Lags to Analyze", 5, 40, 20, 1)
            ccf_horizon = st.slider("Prediction Horizon for CCF (Days)", 1, 20, 5, 1)

            if selected_features:
                st.subheader("Cross-Correlation Results")

                ccf_results = multi_feature_ccf_analysis(
                    data, selected_features, 'Close', nlags=ccf_nlags, target_horizon=ccf_horizon
                )

                for feature in selected_features:
                    if feature in ccf_results:
                        crosscorr, insights = ccf_results[feature]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"{feature} - Best Lag", f"{insights['best_lag']} days")
                        with col2:
                            st.metric(f"{feature} - Best Correlation", f"{insights['best_correlation']:.3f}")

                        with st.expander(f"Show details for {feature}"):
                            st.text(get_summary_statistics('CCF', insights))

                            ccf_fig = plot_ccf(crosscorr, feature, title=f"CCF: {feature} vs Future {ccf_horizon}-Day Returns")
                            st.pyplot(ccf_fig)
                            plt.close(ccf_fig)

                st.divider()

                st.subheader("Feature Engineering Recommendations")
                suggestions = suggest_lag_features(ccf_results, min_correlation=0.1, max_features=10)

                if suggestions:
                    st.info("Based on CCF analysis, these lag features would add to most predictive power:")

                    recommendation_df = pd.DataFrame(suggestions)
                    st.dataframe(recommendation_df)

                    st.success(f"Found {len(suggestions)} features with significant predictive power!")
                else:
                    st.warning("No features met the minimum correlation threshold. Try adjusting parameters or feature selection.")
        else:
            st.warning("No technical features available for CCF analysis. Make sure technical indicators are calculated first.")
    else:
        st.info("Run backtest first to generate feature analysis.")

with tab3:
    st.header("Model Insights")
    st.info("Run backtest first to see model performance insights.")

    if 'model' in st.session_state:
        model = st.session_state['model']
        st.subheader("Feature Importance")

        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': model.feature_name_,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            top_features = feature_importance.head(10)
            st.bar_chart(top_features.set_index('Feature'))

            with st.expander("Show all feature importances"):
                st.dataframe(feature_importance)
        else:
            st.warning("Feature importance not available for this model.")
