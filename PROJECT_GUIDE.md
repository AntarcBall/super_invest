# 📈 AI Algorithmic Trading System - Complete Guide

This project is a sophisticated **AI-driven trading bot** that combines Machine Learning (LightGBM), Statistical Regime Detection (HMM), and Fundamental Analysis (Earnings Surprises) to make intelligent trading decisions.

It features an **Intelligent Agent** that doesn't just predict "Buy/Sell" but calculates a **Conviction Score** based on multiple market factors, optimizing its own parameters using Bayesian Optimization.

---

## 🏗️ System Architecture

The pipeline consists of 5 distinct layers:

1.  **Data Layer**: Fetches OHLCV (Yahoo), Interest Rates (FRED), and Earnings Surprises (FMP).
2.  **Feature Layer**:
    *   **Technical**: RSI, MACD, Bollinger Bands, Volatility (ATR/VIX).
    *   **Macro**: Federal Funds Rate impact.
    *   **Fundamental**: Earnings shocks with time-decay.
    *   **Regime**: HMM (Hidden Markov Model) to detect Bull/Bear/Sideways markets.
3.  **Prediction Layer**: LightGBM Classifier to predict directional movement.
4.  **Decision Layer (`TradingAgent`)**: Combines ML probabilities with Trend, Momentum, Volatility, and Regime signals into a final score.
5.  **Execution Layer (`Backtester`)**: Simulates trades with dynamic position sizing and volatility scaling.

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10+
- (Optional) Financial Modeling Prep (FMP) API Key for earnings data.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# Key libraries: pandas, numpy, lightgbm, hmmlearn, optuna, yfinance, streamlit
```

### 3. Environment Configuration
Create a `.env` file in the root directory to enable advanced features:

```ini
# .env file
FMP_API_KEY=your_fmp_api_key_here
GOOGLE_API_KEY=your_gemini_key_here  # (Optional, for text sentiment)
```
*Note: If no API key is provided, the system will run with Technical + Macro data only.*

---

## 🏃‍♂️ How to Run

### 1. The Main Pipeline (CLI)
This is the core script. It fetches data, trains the model, runs **Optuna Optimization**, and executes a backtest.

```bash
python main.py
```

**What happens:**
1.  **Data Fetch**: Downloads AAPL data (2020-2024).
2.  **Feature Engineering**: Adds indicators and detects Market Regimes (Bull/Bear).
3.  **Training**: Trains LightGBM on the training split (80%).
4.  **Optimization**: Runs `ParameterOptimizer` (Optuna) to find the best weights (e.g., "Trust Volatility more than Momentum").
5.  **Backtest**: Runs the agent on unseen Test data (2024).
6.  **Report**: Prints Total Return, Sharpe Ratio, and Drawdown.

### 2. The Interactive Dashboard (Frontend)
Visualize the strategy, explore data, and run simulations using Streamlit.

```bash
streamlit run backtest_app.py
```
*Access it at `http://localhost:8501` in your browser.*

---

## 🧠 The "Intelligent Agent" Logic

Unlike simple bots that buy when `Prediction > 0.5`, this Agent calculates a weighted **Conviction Score (0 to 1)**:

| Factor | Weight (Optimizable) | Logic |
| :--- | :--- | :--- |
| **ML Model** | ~50% | Probability of price rise from LightGBM. |
| **Trend** | ~15% | Is Price > SMA50 > SMA200? (Bullish alignment). |
| **Momentum** | ~10% | RSI dip buying in uptrends. |
| **Volatility** | ~15% | Low VIX/ATR = Higher confidence. |
| **Macro** | ~5% | Lower Interest Rates = Higher equity allocation. |
| **Earnings** | ~5% | Recent positive earnings surprise = Boost score. |

**Regime Overlay**:
- **Bull Market (HMM)**: Multiplies score by **1.2x** (Aggressive).
- **Bear Market (HMM)**: Multiplies score by **0.5x** (Defensive).

**Dynamic Sizing**:
- The Agent increases position size as Conviction increases.
- It **cuts position size** if Market Volatility (ATR) spikes (Risk Management).

---

## ⚙️ Customization

### Modifying the Strategy
Open `main.py` to adjust the configuration:

```python
# main.py

TICKER = 'TSLA'        # Change stock
HORIZON = 5            # Predict 5 days ahead
DO_OPTIMIZATION = True # Set False to use manual weights
```

### Tuning the Agent
Open `src/backtesting/agent.py` to change the logic rules or `src/backtesting/optimizer.py` to change the search space for parameters.

---

## 📂 File Structure

```
.
├── main.py                  # Entry point for training & backtesting
├── backtest_app.py          # Streamlit Dashboard
├── src/
│   ├── data/
│   │   ├── data_loader.py       # Yahoo Finance fetcher
│   │   ├── macro_loader.py      # FRED Interest Rates
│   │   └── earnings_loader.py   # FMP Earnings Surprises
│   ├── features/
│   │   ├── feature_builder.py   # Technical indicators
│   │   └── regime.py            # HMM Regime Detection
│   ├── models/
│   │   ├── train_model.py       # LightGBM training
│   │   └── lstm_model.py        # (Legacy) LSTM implementation
│   └── backtesting/
│       ├── agent.py             # The Intelligent Trading Agent class
│       ├── engine.py            # Backtest simulation loop
│       └── optimizer.py         # Optuna parameter tuning
└── requirements.txt
```
