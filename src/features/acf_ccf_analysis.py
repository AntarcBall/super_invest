"""
ACF/CCF Analysis Module for Time Series Feature Engineering

This module provides functions to:
1. Analyze Autocorrelation Function (ACF) of stock prices
2. Analyze Cross-Correlation Function (CCF) between features and target
3. Generate visualizations for ACF/CCF analysis
4. Suggest optimal lag features based on statistical significance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, ccf
from typing import Dict, List, Tuple
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)


def analyze_acf(
    df: pd.DataFrame,
    column: str = 'Close',
    nlags: int = 40,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Perform Autocorrelation Function (ACF) analysis on a time series.

    Args:
        df: DataFrame containing the time series
        column: Name of the column to analyze
        nlags: Number of lags to include in ACF
        alpha: Significance level for confidence intervals

    Returns:
        autocorr: Array of autocorrelation coefficients
        confint: Confidence intervals for autocorrelation
        insights: Dictionary with analysis insights
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    series = df[column].dropna()

    autocorr, confint = acf(series, nlags=nlags, alpha=alpha, fft=True)

    lower_bound = confint[:, 0] - autocorr
    upper_bound = confint[:, 1] - autocorr

    significant_lags = []
    for lag in range(1, nlags + 1):
        if abs(autocorr[lag]) > abs(upper_bound[lag]):
            significant_lags.append(lag)
    insights = {
        'significant_lags': significant_lags,
        'autocorr_at_lag1': autocorr[1],
        'autocorr_at_lag5': autocorr[5] if len(autocorr) > 5 else None,
        'autocorr_at_lag10': autocorr[10] if len(autocorr) > 10 else None,
        'strongest_lag': int(np.argmax(np.abs(autocorr[1:])) + 1),
        'trend_strength': autocorr[1] if len(autocorr) > 1 else 0
    }

    return autocorr, confint, insights


def analyze_ccf(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str = 'Close',
    nlags: int = 20,
    target_horizon: int = 5
) -> Tuple[np.ndarray, Dict]:
    """
    Perform Cross-Correlation Function (CCF) analysis between a feature and target.

    Args:
        df: DataFrame containing feature and target
        feature_col: Name of the feature column
        target_col: Name of the target column
        nlags: Number of lags to analyze
        target_horizon: How many days ahead to predict

    Returns:
        crosscorr: Array of cross-correlation coefficients
        insights: Dictionary with analysis insights
    """
    if feature_col not in df.columns:
        raise ValueError(f"Feature column '{feature_col}' not found in DataFrame")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    future_returns = df[target_col].shift(-target_horizon).pct_change(target_horizon)
    feature_data = df[feature_col].dropna()

    aligned_df = pd.DataFrame({
        'feature': feature_data,
        'target': future_returns
    }).dropna()

    crosscorr = np.zeros(nlags + 1)
    for lag in range(nlags + 1):
        if lag == 0:
            crosscorr[lag] = aligned_df['feature'].corr(aligned_df['target'])
        else:
            lagged_feature = aligned_df['feature'].shift(lag)
            crosscorr[lag] = lagged_feature.corr(aligned_df['target'])
    predictive_lags = []
    for lag in range(1, nlags + 1):
        if not np.isnan(crosscorr[lag]) and abs(crosscorr[lag]) > 0.1:
            predictive_lags.append(lag)

    best_lag = np.argmax(np.abs(crosscorr[1:])) + 1 if len(predictive_lags) > 0 else 0
    best_corr = crosscorr[best_lag] if best_lag > 0 else 0

    insights = {
        'predictive_lags': predictive_lags,
        'best_lag': best_lag,
        'best_correlation': best_corr,
        'correlation_at_lag1': crosscorr[1] if len(crosscorr) > 1 else None,
        'correlation_at_lag5': crosscorr[5] if len(crosscorr) > 5 else None,
        'predictive_power': 'high' if abs(best_corr) > 0.3 else 'medium' if abs(best_corr) > 0.15 else 'low'
    }

    return crosscorr, insights


def plot_acf(
    autocorr: np.ndarray,
    confint: np.ndarray,
    title: str = "Autocorrelation Function (ACF)",
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Create ACF visualization plot.

    Args:
        autocorr: Array of autocorrelation coefficients
        confint: Confidence intervals
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    nlags = len(autocorr) - 1
    lags = np.arange(nlags + 1)

    ax.bar(lags, autocorr, width=0.3, color='steelblue', alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    for lag in range(nlags + 1):
        ci_lower = confint[lag, 0]
        ci_upper = confint[lag, 1]
        ax.plot([lag, lag], [ci_lower, ci_upper], color='red', alpha=0.6, linewidth=1.5)

    significant_lags = []
    lower_bound = confint[:, 0] - autocorr
    upper_bound = confint[:, 1] - autocorr
    for lag in range(1, nlags + 1):
        if abs(autocorr[lag]) > abs(upper_bound[lag]):
            ax.bar(lag, autocorr[lag], width=0.3, color='coral', alpha=0.9)
            significant_lags.append(lag)

    ax.set_xlabel('Lag (Days)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if significant_lags:
        ax.text(0.02, 0.98, f'Significant Lags: {significant_lags}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_ccf(
    crosscorr: np.ndarray,
    feature_name: str,
    title: str = "Cross-Correlation Function (CCF)",
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Create CCF visualization plot.

    Args:
        crosscorr: Array of cross-correlation coefficients
        feature_name: Name of the feature being analyzed
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    nlags = len(crosscorr) - 1
    lags = np.arange(nlags + 1)

    colors = ['steelblue' if x >= 0 else 'coral' for x in crosscorr]
    ax.bar(lags, crosscorr, width=0.3, color=colors, alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    for lag, corr in enumerate(crosscorr):
            if abs(corr) > 0.2:
                ax.text(lag, corr, f'{corr:.2f}', ha='center', va='bottom' if corr > 0 else 'top',
                       fontsize=8, fontweight='bold')

    ax.set_xlabel('Lag (Days)')
    ax.set_ylabel('Cross-Correlation with Future Returns')
    ax.set_title(f'{title}\nFeature: {feature_name}')
    ax.grid(True, alpha=0.3)

    best_lag = np.argmax(np.abs(crosscorr[1:])) + 1 if len(crosscorr) > 1 else 0
    if best_lag > 0:
        ax.axvline(x=best_lag, color='green', linestyle='--', alpha=0.5, linewidth=2)
        ax.text(best_lag, ax.get_ylim()[1] * 0.9, f'Best Lag: {best_lag}',
               ha='center', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    return fig


def multi_feature_ccf_analysis(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'Close',
    nlags: int = 20,
    target_horizon: int = 5
) -> Dict[str, Tuple[np.ndarray, Dict]]:
    """
    Perform CCF analysis on multiple features at once.

    Args:
        df: DataFrame containing features and target
        feature_cols: List of feature column names to analyze
        target_col: Name of the target column
        nlags: Number of lags to analyze
        target_horizon: How many days ahead to predict

    Returns:
        Dictionary mapping feature names to their CCF results and insights
    """
    results = {}

    for feature in feature_cols:
        if feature in df.columns:
            try:
                crosscorr, insights = analyze_ccf(
                    df, feature, target_col, nlags, target_horizon
                )
                results[feature] = (crosscorr, insights)
            except Exception as e:
                print(f"ERROR analyzing {feature}: {e}")
                continue

    return results


def suggest_lag_features(
    ccf_results: Dict[str, Tuple[np.ndarray, Dict]],
    min_correlation: float = 0.15,
    max_features: int = 10
) -> List[Dict[str, any]]:
    """
    Suggest lag features based on CCF analysis results.

    Args:
        ccf_results: Dictionary of CCF analysis results from multi_feature_ccf_analysis
        min_correlation: Minimum absolute correlation to consider
        max_features: Maximum number of features to suggest

    Returns:
        List of dictionaries with suggested features (sorted by correlation strength)
    """
    suggestions = []

    for feature_name, (crosscorr, insights) in ccf_results.items():
        if insights['best_correlation'] is not None and abs(insights['best_correlation']) >= min_correlation:
            suggestions.append({
                'feature': feature_name,
                'lag': insights['best_lag'],
                'correlation': insights['best_correlation'],
                'predictive_power': insights['predictive_power']
            })

    suggestions.sort(key=lambda x: abs(x['correlation']), reverse=True)

    return suggestions[:max_features]


def get_summary_statistics(analysis_type: str, insights: Dict) -> str:
    """
    Generate human-readable summary of ACF/CCF analysis results.

    Args:
        analysis_type: 'ACF' or 'CCF'
        insights: Dictionary of analysis insights

    Returns:
        Formatted summary string
    """
    if analysis_type == 'ACF':
        summary = f"""
ACF Analysis Summary:
- Trend Strength (Lag 1): {insights['trend_strength']:.3f}
- Significant Lags: {insights['significant_lags']}
- Strongest Lag: {insights['strongest_lag']} days
- Autocorr at Lag 5: {f"{insights['autocorr_at_lag5']:.3f}" if insights['autocorr_at_lag5'] else 'N/A'}
"""
    elif analysis_type == 'CCF':
        summary = f"""
CCF Analysis Summary:
- Best Predictive Lag: {insights['best_lag']} days
- Best Correlation: {insights['best_correlation']:.3f}
- Predictive Power: {insights['predictive_power']}
- All Predictive Lags: {insights['predictive_lags']}
"""
    else:
        summary = "Unknown analysis type"

    return summary


if __name__ == '__main__':
    from src.data.data_loader import get_stock_data
    from src.features.feature_builder import add_technical_features, add_volatility_features

    print("Testing ACF/CCF Analysis Module...")
    data = get_stock_data('AAPL', '2023-01-01', '2024-01-01')

    if not data.empty:
        data = add_technical_features(data)
        data = add_volatility_features(data)

        print("\n--- Testing ACF Analysis ---")
        autocorr, confint, acf_insights = analyze_acf(data, 'Close', nlags=40)
        print(get_summary_statistics('ACF', acf_insights))

        print("\n--- Testing CCF Analysis ---")
        features_to_test = ['RSI_14', 'MACD_12_26_9', 'ATR_14', 'VIX']
        for feature in features_to_test:
            if feature in data.columns:
                crosscorr, ccf_insights = analyze_ccf(data, feature, 'Close', nlags=20)
                print(f"\nFeature: {feature}")
                print(get_summary_statistics('CCF', ccf_insights))

        print("\n--- Testing Multi-Feature CCF Analysis ---")
        ccf_results = multi_feature_ccf_analysis(data, features_to_test, 'Close', nlags=20)
        suggestions = suggest_lag_features(ccf_results)

        print("\nSuggested Lag Features:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion['feature']}_lag_{suggestion['lag']} (Corr: {suggestion['correlation']:.3f})")

        print("\n✓ ACF/CCF Analysis Module test completed successfully!")
    else:
        print("ERROR: Could not load test data")
