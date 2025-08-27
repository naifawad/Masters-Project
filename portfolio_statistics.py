# === portfolio_statistics.py (Fixed for Monthly Returns) ===
import numpy as np
import pandas as pd

def _to_series(x):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            raise ValueError("Expected 1D input (Series or 1-column DataFrame).")
    return pd.Series(x).dropna().astype(float)


def sharpe_ratio_monthly(returns, risk_free_rate):
    r = _to_series(returns)
    rf_monthly = risk_free_rate / 12.0
    excess = r - rf_monthly
    vol = excess.std(ddof=1) * np.sqrt(12)
    if not np.isfinite(vol) or vol == 0:
        return np.nan
    return float(excess.mean() * 12 / vol)


def sortino_ratio_monthly(returns, risk_free_rate, mar=None, eps=1e-12):
    r = _to_series(returns)
    rf_monthly = (mar / 12.0) if mar is not None else (risk_free_rate / 12.0)
    excess = r - rf_monthly
    downside = excess[excess < 0]

    denom = downside.std(ddof=1) * np.sqrt(12)
    if not np.isfinite(denom) or denom < eps:
        return np.nan
    return float(excess.mean() * 12 / denom)


def information_ratio_monthly(returns, benchmark_returns):
    r = _to_series(returns)
    b = _to_series(benchmark_returns)
    r, b = r.align(b, join='inner')
    active = r - b
    vol = active.std(ddof=1) * np.sqrt(12)
    if not np.isfinite(vol) or vol == 0:
        return np.nan
    return float(active.mean() * 12 / vol)


def annualized_volatility_monthly(returns):
    r = _to_series(returns)
    return float(r.std(ddof=1) * np.sqrt(12))


def max_drawdown(series):
    # Handle DataFrame or array inputs safely
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 1:
            series = series.iloc[:, 0]
        else:
            raise ValueError(f"Expected 1D series or 1-column DataFrame. Got shape: {series.shape}")
    elif isinstance(series, (np.ndarray, list)):
        series = pd.Series(series)
    
    s = pd.Series(series).dropna()
    peak = s.cummax()
    drawdown = (s - peak) / peak
    return float(drawdown.min())



def calculate_correlation_with_market(returns, market_returns):
    return returns.corr(market_returns)


def calculate_turnover(weights_df):
    # Turnover is the sum of absolute weight changes between rebalances
    turnover = weights_df.diff().abs().sum(axis=1)
    return turnover.mean()
