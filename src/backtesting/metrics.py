import numpy as np
import pandas as pd

def calculate_sharpe_ratio(portfolio_df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """
    Calculates the annualized Sharpe ratio of a strategy.

    Args:
        portfolio_df (pd.DataFrame): DataFrame with 'PortfolioValue' column.
        risk_free_rate (float): The annual risk-free rate.

    Returns:
        float: The annualized Sharpe ratio.
    """
    daily_returns = portfolio_df['PortfolioValue'].pct_change().dropna()
    
    # Assuming 252 trading days in a year
    annualized_return = daily_returns.mean() * 252
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    
    if annualized_volatility == 0:
        return 0.0

    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    return sharpe_ratio

def calculate_max_drawdown(portfolio_df: pd.DataFrame) -> float:
    """
    Calculates the maximum drawdown of a strategy.

    Args:
        portfolio_df (pd.DataFrame): DataFrame with 'PortfolioValue' column.

    Returns:
        float: The maximum drawdown as a percentage.
    """
    cumulative_max = portfolio_df['PortfolioValue'].cummax()
    drawdown = (portfolio_df['PortfolioValue'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    return max_drawdown * 100

if __name__ == '__main__':
    # Example Usage:
    dates = pd.date_range('2023-01-01', periods=100)
    values = 100000 + pd.Series(np.random.randn(100) * 100).cumsum()
    values[30:50] -= 2000 # Introduce a drawdown
    
    example_df = pd.DataFrame({'PortfolioValue': values}, index=dates)
    
    sharpe = calculate_sharpe_ratio(example_df, risk_free_rate=0.02)
    max_dd = calculate_max_drawdown(example_df)
    
    print("--- Performance Metrics Example ---")
    print(f"Annualized Sharpe Ratio: {sharpe:.2f}")
    print(f"Maximum Drawdown: {max_dd:.2f}%")

