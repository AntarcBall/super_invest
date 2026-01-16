import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data.data_loader import get_stock_data
from src.backtesting.agent import TradingAgent
from src.backtesting.optimizer import ParameterOptimizer
from src.features.regime import RegimeDetector
from src.data.earnings_loader import EarningsLoader
from src.features.feature_builder import add_fundamental_features, add_technical_features, add_macro_features, add_volatility_features, add_lagged_features, add_acf_ccf_lagged_features, add_earnings_features
from src.features.acf_ccf_analysis import analyze_acf, analyze_ccf, multi_feature_ccf_analysis, suggest_lag_features, plot_acf, plot_ccf, get_summary_statistics
from src.models.prepare_data import prepare_training_data
from src.models.train_model import train_lgbm_model
from src.models.pretrained_utils import add_pretrained_embeddings, load_pretrained_model, extract_embeddings_from_stock
from src.backtesting.engine import Backtester
from src.backtesting.metrics import calculate_sharpe_ratio, calculate_max_drawdown
import datetime
import os

@st.cache_resource
def get_pretrained_model(model_path, config_path):
    return load_pretrained_model(model_path, config_path)

st.set_page_config(layout="wide", page_title="Stock Prediction & Backtesting Engine", page_icon="📈")

st.title("📈 Stock Prediction & Backtesting Engine")
st.markdown("""
**Enhanced with Transfer Learning from 91 Major Stocks**
*GPU-accelerated pretraining with RTX 3090, advanced ACF/CCF feature engineering*
""")

tab1, tab2, tab3, tab4 = st.tabs(["Backtest", "Feature Analysis", "Model Insights", "System Info"])

with st.sidebar:
    st.header("⚙️ Configuration")

    ticker = st.text_input("Stock Ticker", "AAPL")
    st.caption("Enter stock symbol (e.g., AAPL, MSFT, GOOGL)")

    today = datetime.date.today()
    start_date = st.date_input("Start Date", today - datetime.timedelta(days=3*365))
    end_date = st.date_input("End Date", today)

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    horizon = st.slider("Prediction Horizon (Days)", 1, 30, 10, 1)
    threshold = st.slider("Buy Signal Threshold (%)", 1.0, 10.0, 3.0, 0.5) / 100.0

    st.divider()

    st.header("🧠 Model Options")
    
    st.subheader("Data Sources")
    use_earnings = st.checkbox("Include Earnings Surprises (FMP)", value=True, help="Requires FMP_API_KEY")
    use_macro = st.checkbox("Include Macro Data (FRED)", value=True, help="Federal Funds Rate")
    use_volatility = st.checkbox("Include Volatility (VIX/ATR)", value=True)
    use_regime = st.checkbox("Use Regime Detection (HMM)", value=True, help="Detect Bull/Bear markets")
    
    st.divider()
    
    use_acf_ccf_features = st.checkbox("Use ACF/CCF-Based Lag Features", value=True, help="Enhanced feature engineering using statistical analysis")
    use_pretrained_embeddings = st.checkbox("Use Pretrained Market Embeddings", value=False, help="Use embeddings from model trained on 91 major stocks (requires pretrained model)")

    if use_pretrained_embeddings:
        st.info("💡 Using transfer learning from 91 major stocks (2019-2024)")
        st.success("🚀 Enhanced with bidirectional LSTM + batch normalization")
        st.success("⚡ Mixed precision training enabled")

    run_button = st.button("🚀 Run Backtest", type="primary")

with tab2:
    st.header("🤖 Model Training & Optimization")
    st.info("Optimize the Intelligent Agent's parameters using Bayesian Optimization (Optuna).")
    
    col1, col2 = st.columns(2)
    with col1:
        n_trials = st.slider("Optimization Trials", 5, 100, 20, 5, help="More trials = better results but slower")
    with col2:
        opt_initial_cash = st.number_input("Initial Cash for Optimization", value=100000)

    if st.button("🚀 Start Optimization Loop", type="primary"):
        with st.status("Running Optimization...", expanded=True) as status:
            # 1. Load & Prepare Data
            status.update(label="Loading and preparing data...")
            data = get_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            
            if data.empty:
                st.error("No data found.")
                status.update(state="error")
            else:
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                
                # Feature Engineering based on flags
                data = add_fundamental_features(data, ticker)
                data = add_technical_features(data)
                
                if use_macro:
                    data = add_macro_features(data)
                if use_volatility:
                    data = add_volatility_features(data)
                if use_earnings:
                    data = add_earnings_features(data, ticker)
                
                if use_acf_ccf_features:
                    data = add_acf_ccf_lagged_features(data, target_horizon=horizon)
                else:
                    data = add_lagged_features(data)
                    
                if use_regime:
                    status.update(label="Detecting Market Regimes...")
                    rd = RegimeDetector()
                    rd.fit(data)
                    regimes = rd.predict_regime(data)
                    data = data.join(regimes)
                    data['Regime'] = data['Regime'].ffill().bfill()

                # Prepare Training Data
                X, y = prepare_training_data(data, horizon=horizon, threshold=threshold)
                
                # Split (Train only for optimization)
                split_index = int(len(X) * (1 - test_size))
                X_train = X.iloc[:split_index]
                y_train = y[:split_index]
                
                status.update(label="Training Base Model...")
                model = train_lgbm_model(X_train, y_train, X_train, y_train) # Eval on train for speed/sanity
                
                status.update(label=f"Running Optuna ({n_trials} trials)...")
                optimizer = ParameterOptimizer(model, X_train, y_train, opt_initial_cash)
                best_params = optimizer.optimize_params(n_trials=n_trials)
                
                status.update(label="Optimization Complete!", state="complete")
                
                # Save to session state
                st.session_state['optimized_params'] = best_params
                st.session_state['trained_model'] = model
                st.session_state['feature_data'] = data # Reuse data
                
                st.success(f"Optimization finished! Best Sharpe: {best_params.get('sharpe', 'N/A')}")
                
                # Display Results
                st.subheader("🏆 Optimized Parameters")
                
                # Format weights for display
                weights = {k: v for k, v in best_params.items() if k.startswith('w_')}
                # Normalize
                total = sum(weights.values())
                if total > 0:
                    weights = {k: v/total for k, v in weights.items()}
                
                thresholds = {k: v for k, v in best_params.items() if k.startswith('thresh_')}
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.caption("Agent Weights")
                    st.json(weights)
                    st.bar_chart(pd.Series(weights))
                with res_col2:
                    st.caption("Decision Thresholds")
                    st.json(thresholds)
                
                st.info("These parameters are now saved and will be used in the 'Backtest' tab.")

with tab1:
    st.header("📊 Backtesting Engine")

    if use_pretrained_embeddings:
        st.success("🚀 **TRANSFER LEARNING ENABLED**: Using embeddings from model trained on 91 major stocks")
        st.info("Enhanced with bidirectional LSTM, batch normalization, and mixed precision training")
    else:
        st.info("ℹ️ **STANDARD MODE**: Using only technical/fundamental features without transfer learning")

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
                
                # Feature Engineering based on flags
                data = add_fundamental_features(data, ticker)
                data = add_technical_features(data)
                
                if use_macro:
                    data = add_macro_features(data)
                if use_volatility:
                    data = add_volatility_features(data)
                if use_earnings:
                    data = add_earnings_features(data, ticker)
                
                status.update(label="Step 3/6: Adding Lagged Features...")
                if use_acf_ccf_features:
                    data = add_acf_ccf_lagged_features(data, target_horizon=horizon)
                else:
                    data = add_lagged_features(data)
                    
                if use_regime:
                    status.update(label="Detecting Market Regimes...")
                    rd = RegimeDetector()
                    rd.fit(data)
                    regimes = rd.predict_regime(data)
                    data = data.join(regimes)
                    data['Regime'] = data['Regime'].ffill().bfill()

                status.update(label="Step 4/6: Preparing Training Data...")
                X, y = prepare_training_data(data, horizon=horizon, threshold=threshold)

                if use_pretrained_embeddings:
                    status.update(label="Step 4.5/6: Adding Pretrained Embeddings...")
                    try:
                        model_pth = 'models/pretrained_lstm.pth'
                        config_json = 'models/pretrained_config.json'

                        if not os.path.exists(model_pth):
                            st.warning("⚠️ Pretrained model file not found. Skipping embeddings. Run `bash run_pretrain.sh` to generate model.")
                        else:
                            pretrained_model, device = get_pretrained_model(model_pth, config_json)
                            split_index = int(len(X) * (1 - test_size))
                            X_test_only = X.iloc[split_index:]

                            # Display model info
                            st.info(f"Loading pretrained model on {device}")
                            st.info(f"  - Input dimension: {getattr(pretrained_model, 'input_dim', 'Unknown')}")
                            st.info(f"  - Hidden dimension: {getattr(pretrained_model, 'hidden_dim', 'Unknown')}")
                            st.info(f"  - Embedding dimension: {getattr(pretrained_model, 'embedding_dim', 'Unknown')}")

                            embedding_df = extract_embeddings_from_stock(pretrained_model, X_test_only, device)

                            if embedding_df is not None:
                                X_combined = pd.concat([X_test_only.loc[embedding_df.index], embedding_df], axis=1)
                                X_test_final = X_combined.sort_index()
                                y_test_final = y.loc[X_test_final.index]

                                st.success(f"✅ Added {len(embedding_df.columns)} market embedding features.")
                                st.info(f"Test data dimension: {X_test_final.shape[0]} samples, {X_test_final.shape[1]} features (including embeddings)")
                            else:
                                st.error("❌ Failed to extract embeddings.")
                    except Exception as e:
                        st.error(f"❌ Error during embedding extraction: {e}")
                else:
                    status.update(label="Step 4.5/6: Pretrained embeddings not enabled. Skipping.")

                status.update(label="Step 5/6: Preparing Training Data...")
                split_index = int(len(X) * (1 - test_size))

                if use_pretrained_embeddings and 'X_test_final' in locals():
                    X_train = X.iloc[:split_index]
                    X_test = X_test_final
                    y_train = y_test[:split_index]
                    y_test = y_test_final
                else:
                    X_train, X_test = X[:split_index], X[split_index:]
                    y_train, y_test = y[:split_index], y[split_index:]

                st.info(f"Training model with {X_train.shape[1]} features on {X_train.shape[0]} samples")

                model = train_lgbm_model(X_train, y_train, X_test, y_test)

                status.update(label="Step 6/6: Running Backtest...")
                
                # Check for optimized params
                if 'optimized_params' in st.session_state:
                    params = st.session_state['optimized_params']
                    st.success("Using Optimized Parameters from Training Tab")
                    weights = {
                        'model': params['w_model'],
                        'trend': params['w_trend'],
                        'momentum': params['w_momentum'],
                        'volatility': params['w_volatility'],
                        'macro': params['w_macro'],
                        'sentiment': 0.05 # Default if not optimized
                    }
                    # Normalize
                    tot = sum(weights.values())
                    if tot > 0: weights = {k: v/tot for k, v in weights.items()}
                    
                    thresholds = {
                        'buy': params['thresh_buy'],
                        'sell': params['thresh_sell'],
                        'strong_buy': params['thresh_buy'] + 0.15,
                        'strong_sell': params['thresh_sell'] - 0.15
                    }
                    agent = TradingAgent(model, weights=weights, thresholds=thresholds)
                else:
                    st.info("Using Default Parameters (Run Optimization in Tab 2 to improve)")
                    agent = TradingAgent(model) # Defaults
                
                backtester = Backtester(agent, X_test, y_test)
                portfolio_df = backtester.run_backtest()

                status.update(label="Pipeline complete!", state="complete")


                st.success("✅ Backtest finished. See results below.")

                st.session_state['data'] = data
                st.session_state['model'] = model
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test

                st.header("📈 Performance Analysis")

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

                # Calculate improvement metrics
                return_improvement = total_return_strategy - total_return_benchmark
                sharpe_improvement = sharpe_strategy - sharpe_benchmark

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Our Strategy")
                    st.metric("Total Return", f"{total_return_strategy:.2f}%",
                             f"{return_improvement:+.2f}% vs Buy&Hold")
                    st.metric("Sharpe Ratio", f"{sharpe_strategy:.2f}",
                             f"{sharpe_improvement:+.2f} vs Buy&Hold")
                    st.metric("Max Drawdown", f"{max_drawdown_strategy:.2f}%")
                    st.metric("Final Value", f"${final_value_strategy:,.2f}")

                with col2:
                    st.subheader("Buy & Hold Benchmark")
                    st.metric("Total Return", f"{total_return_benchmark:.2f}%")
                    st.metric("Sharpe Ratio", f"{sharpe_benchmark:.2f}")
                    st.metric("Max Drawdown", f"{max_drawdown_benchmark:.2f}%")
                    st.metric("Final Value", f"${(100000 * (1 + total_return_benchmark/100)):,.2f}")

                # Add improvement summary
                if use_pretrained_embeddings:
                    st.subheader("🎯 Transfer Learning Impact")
                    improvement_col1, improvement_col2 = st.columns(2)
                    with improvement_col1:
                        if return_improvement > 0:
                            st.success(f"Return improvement: +{return_improvement:.2f}%")
                        else:
                            st.warning(f"Return difference: {return_improvement:.2f}%")

                    with improvement_col2:
                        if sharpe_improvement > 0:
                            st.success(f"Sharpe improvement: +{sharpe_improvement:.2f}")
                        else:
                            st.warning(f"Sharpe difference: {sharpe_improvement:.2f}")

                st.header("💰 Portfolio Value Over Time")
                chart_df = pd.DataFrame({
                    'Strategy': portfolio_df['PortfolioValue'],
                    'Buy & Hold': benchmark_portfolio['PortfolioValue']
                })
                st.line_chart(chart_df)

                with st.expander("🔍 Show Final Data with Features"):
                    st.dataframe(data.tail())

with tab2:
    st.header("🔬 Feature Analysis (ACF/CCF)")
    st.info("Statistical analysis to identify temporal patterns and predictive features")

    if 'data' in st.session_state:
        data = st.session_state['data']

        st.subheader("Autocorrelation Analysis (ACF)")
        st.markdown("""
        **ACF (Autocorrelation Function)** helps identify:
        - Temporal patterns in stock prices
        - Trend strength and persistence
        - Optimal lag periods for feature engineering
        """)

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
        st.markdown("""
        **CCF (Cross-Correlation Function)** identifies:
        - Which features are most predictive of future returns
        - Optimal time lags for each feature
        - Feature engineering opportunities
        """)

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

                st.subheader("💡 Feature Engineering Recommendations")
                suggestions = suggest_lag_features(ccf_results, min_correlation=0.1, max_features=10)

                if suggestions:
                    st.info("Based on CCF analysis, these lag features would add the most predictive power:")

                    recommendation_df = pd.DataFrame(suggestions)
                    st.dataframe(recommendation_df)

                    st.success(f"✅ Found {len(suggestions)} features with significant predictive power!")

                    with st.expander("How to use these recommendations"):
                        st.markdown("""
                        **Next Steps:**
                        1. Add these lag features to your model
                        2. Retrain with the enhanced feature set
                        3. Compare performance with baseline
                        4. Use in combination with transfer learning embeddings
                        """)
                else:
                    st.warning("No features met the minimum correlation threshold. Try adjusting parameters or feature selection.")
        else:
            st.warning("No technical features available for CCF analysis. Make sure technical indicators are calculated first.")
    else:
        st.info("Run backtest first to generate feature analysis.")

with tab3:
    st.header("🧠 Model Insights")
    st.info("Run backtest first to see model performance insights.")

    if 'model' in st.session_state:
        model = st.session_state['model']
        st.subheader("Feature Importance Analysis")

        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': model.feature_name_,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            # Count how many are embedding features
            embedding_features = [f for f in feature_importance['Feature'] if 'lstm_emb_' in f]
            regular_features = [f for f in feature_importance['Feature'] if 'lstm_emb_' not in f]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Features", len(feature_importance))
            with col2:
                st.metric("Embedding Features", len(embedding_features))

            if embedding_features:
                st.success(f"✅ Transfer learning embeddings are contributing to predictions ({len(embedding_features)} features)")

            top_features = feature_importance.head(10)
            st.bar_chart(top_features.set_index('Feature'))

            with st.expander("Show all feature importances"):
                st.dataframe(feature_importance)

            with st.expander("Embedding Feature Analysis"):
                if embedding_features:
                    embedding_importance = feature_importance[feature_importance['Feature'].isin(embedding_features)]
                    st.write(f"Embedding features contribute {embedding_importance['Importance'].sum()*100:.2f}% of total importance")
                    st.bar_chart(embedding_importance.set_index('Feature'))
                else:
                    st.info("No embedding features detected - make sure 'Use Pretrained Market Embeddings' is enabled")
        else:
            st.warning("Feature importance not available for this model.")

        st.divider()

        st.subheader("Model Performance Metrics")
        if 'X_test' in st.session_state and 'y_test' in st.session_state:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']

            y_pred = model.predict(X_test)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            with col2:
                st.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
            with col3:
                st.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
            with col4:
                st.metric("F1-Score", f"{f1_score(y_test, y_pred, zero_division=0):.3f}")

with tab4:
    st.header("System Information & Model Details")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hardware Acceleration")
        st.markdown("""
        - **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
        - **CUDA Version**: 12.6
        - **Driver Version**: 560.94
        - **Mixed Precision**: Enabled for faster training
        """)

        st.subheader("Model Architecture")
        st.markdown("""
        - **LSTM Layers**: 2 (bidirectional)
        - **Hidden Dimension**: 256
        - **Embedding Dimension**: 128
        - **Dropout**: 0.2
        - **Batch Normalization**: Enabled
        - **Total Parameters**: ~2.3M
        """)

    with col2:
        st.subheader("Transfer Learning Details")
        st.markdown("""
        - **Pretrained Stocks**: 91 major stocks (2019-2024)
        - **Sectors Covered**: Tech, Finance, Healthcare, Energy, Consumer, Industrial
        - **Features Used**: 26+ fundamental, technical, macroeconomic, volatility
        - **Sequence Length**: 20 days
        - **Embedding Type**: 128-dimensional market-aware vectors
        """)

        st.subheader("Training Enhancements")
        st.markdown("""
        - **Optimizer**: AdamW with weight decay
        - **Learning Rate**: Cyclical (0.0001-0.01)
        - **Batch Size**: 128 (reduced for better generalization)
        - **Epochs**: 100 (increased for more training)
        - **Gradient Clipping**: 5.0 (increased for stability)
        """)

    st.divider()

    st.subheader("Model Status")
    model_path = 'models/pretrained_lstm.pth'
    if os.path.exists(model_path):
        st.success("✅ Pretrained model available")
        # Show model info if it exists
        try:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'config' in checkpoint:
                config = checkpoint['config']
                st.info(f"Model trained on {config.get('total_stocks_trained', 'unknown')} stocks")
        except:
            st.info("Model file exists but details unavailable")
    else:
        st.warning("⚠️ Pretrained model not found - run `bash run_pretrain.sh`")

    st.subheader("Quick Start Guide")
    st.code("""
# 1. Train pretrained model (one-time, 2-4 hours)
bash run_pretrain.sh

# 2. Start frontend
bash run_app.sh

# 3. Enable embeddings in sidebar
# 4. Run backtest
    """, language='bash')
