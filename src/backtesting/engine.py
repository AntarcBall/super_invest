import pandas as pd
import numpy as np

class Backtester:
    """
    A simple event-driven backtester for a machine learning trading strategy.
    """
    def __init__(self, model, X_test, y_test, initial_cash=100000):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.portfolio_history = []

    def run_backtest(self):
        """
        Runs the backtesting simulation.
        """
        print("\n--- Running Backtest ---")
        for i, (date, features) in enumerate(self.X_test.iterrows()):
            signal = self.model.predict(features.to_frame().T)[0]
            
            current_price = self.X_test.iloc[i]['Close']
            
            if signal == 1 and self.position == 0:
                self.position = self.cash / current_price
                self.cash = 0
            
            elif signal == 0 and self.position > 0:
                self.cash = self.position * current_price
                self.position = 0
            
            portfolio_value = self.cash + (self.position * current_price)
            self.portfolio_history.append({'Date': self.X_test.index[i], 'PortfolioValue': portfolio_value})
        
        if self.position > 0:
            final_price = self.X_test.iloc[0]['Close']
            self.cash = self.position * final_price
            self.position = 0
            self.portfolio_history[-1]['PortfolioValue'] = self.cash
        
        print("Backtest complete.")
        return pd.DataFrame(self.portfolio_history).set_index('Date')

if __name__ == '__main__':
    from src.data.data_loader import get_stock_data
    from src.features.feature_builder import add_fundamental_features, add_technical_features, add_macro_features
    from src.models.prepare_data import prepare_training_data
    from src.models.train_model import train_lgbm_model
    from src.backtesting.metrics import calculate_sharpe_ratio, calculate_max_drawdown
    from sklearn.model_selection import train_test_split

    TICKER = 'MSFT'
    START = '2022-01-01'
    END = '2024-01-01'
    
    data = get_stock_data(TICKER, START, END)
    
    if not data.empty:
        # Prepare data
        data = add_fundamental_features(data, TICKER)
        data = add_technical_features(data)
        data = add_macro_features(data)
        X, y = prepare_training_data(data, horizon=5, threshold=0.02)
        
        # Split data
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Train model
        model = train_lgbm_model(X, y)
        
        # Run backtest
        backtester = Backtester(model, X_test, y_test)
        portfolio_df = backtester.run_backtest()
        
        print("\n--- Backtest Performance ---")
        
        # Calculate and print metrics
        sharpe_ratio = calculate_sharpe_ratio(portfolio_df, risk_free_rate=0.02)
        max_drawdown = calculate_max_drawdown(portfolio_df)
        
        final_value = portfolio_df['PortfolioValue'].iloc[-1]
        total_return = ((final_value - backtester.initial_cash) / backtester.initial_cash) * 100
        
        print(f"Initial Portfolio Value: ${backtester.initial_cash:,.2f}")
        print(f"Final Portfolio Value:   ${final_value:,.2f}")
        print(f"Total Return:            {total_return:.2f}%")
        print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown:        {max_drawdown:.2f}%")
