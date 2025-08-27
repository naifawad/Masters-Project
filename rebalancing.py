import pandas as pd
import numpy as np
from typing import Callable, List
from mean_variance_optimization import calculate_weights, mean_variance_optimization
from machine_learning_strategies import download_stock_data
from machine_learning_strategies import generate_walkforward_ml_views
from hrp import get_hrp_allocation
from rp import get_risk_parity_weights


# === universal bounds projector (sum=1, lo <= w_i <= hi) ===
import numpy as np
import pandas as pd

def project_box_simplex(v, lo=0.0, hi=0.25, tol=1e-12, max_iter=200):
    v = np.asarray(v, dtype=float)
    n = v.size
    L = np.full(n, float(lo))
    U = np.full(n, float(hi))

    # feasibility: n*lo <= 1 <= n*hi
    if L.sum() - 1.0 > tol:
        raise ValueError(f"Infeasible: n*lo = {L.sum():.4f} > 1; lower bound too high.")
    if U.sum() + tol < 1.0:
        raise ValueError(f"Infeasible: n*hi = {U.sum():.4f} < 1; upper bound too low.")

    # bisection on tau to solve sum( clip(v - tau, L, U) ) = 1
    low_tau  = np.min(v - U)
    high_tau = np.max(v - L)
    for _ in range(max_iter):
        tau = 0.5 * (low_tau + high_tau)
        w = np.clip(v - tau, L, U)
        s = w.sum()
        if abs(s - 1.0) <= tol:
            return w
        if s > 1.0:
            low_tau = tau
        else:
            high_tau = tau
    return np.clip(v - 0.5*(low_tau + high_tau), L, U)

def enforce_bounds_on_df(weights_df: pd.DataFrame, lo=0.0, hi=0.25) -> pd.DataFrame:
    W = weights_df.copy()
    vals = []
    for _, row in W.iterrows():
        vals.append(project_box_simplex(row.values, lo=lo, hi=hi))
    return pd.DataFrame(vals, index=W.index, columns=W.columns)


def get_rebalance_dates(prices: pd.DataFrame, frequency: str) -> List[pd.Timestamp]:
    if frequency == 'M':
        return prices.resample('M').last().index
    elif frequency == 'Q':
        return prices.resample('Q').last().index
    elif frequency == '6M':
        return prices.resample('2Q').last().index
    elif frequency == 'Y':
        return prices.resample('Y').last().index
    else:
        raise ValueError("Invalid frequency. Use 'M', 'Q', '6M', or 'Y'.")

def rebalance_if_deviation(current_weights, target_weights, threshold=0.05):
    delta = np.abs(current_weights - target_weights)
    if np.any(delta > threshold):
        return target_weights
    else:
        return current_weights


def run_custom_rebalancing(prices: pd.DataFrame, weights_fn: Callable[[pd.DataFrame], pd.Series], rebalance_freq: str) -> pd.Series:
    returns = prices.pct_change().dropna()
    rebalance_dates = get_rebalance_dates(prices, rebalance_freq)
    portfolio_value = [1.0]
    dates = [returns.index[0]]
    current_weights = None

    for i in range(1, len(returns)):
        date = returns.index[i]
        prev_value = portfolio_value[-1]
        r = returns.iloc[i]

        if date in rebalance_dates or current_weights is None:
            past_data = prices.loc[:date].dropna()
            current_weights = weights_fn(past_data)
            current_weights = current_weights.reindex(prices.columns).fillna(0)

        daily_return = np.dot(current_weights.values, r.values)
        new_value = prev_value * (1 + daily_return)

        portfolio_value.append(new_value)
        dates.append(date)

    return pd.Series(portfolio_value, index=dates)



def run_monthly_rebalanced_strategies(
    portfolio_dict,
    start_date,
    end_date,
    training_window_years=5,
    max_volatility=0.225,
    min_weight=0.01,
    max_weight=0.25,
    rebalance_freq='M',
):
    tickers, initial_weights = calculate_weights(portfolio_dict)
    assert len(tickers)*min_weight <= 1 <= len(tickers)*max_weight, (
        f"Infeasible bounds: n={len(tickers)}, min={min_weight}, max={max_weight}"
    )


    full_data = download_stock_data(tickers, start_date, end_date)['Adj Close']
    daily_returns_all = full_data.pct_change().dropna()
    unopt_daily_returns = daily_returns_all.dot(initial_weights)
    unopt_cumulative_daily = (1 + unopt_daily_returns).cumprod()
    unopt_cumulative_resampled = unopt_cumulative_daily.resample(rebalance_freq).last()

    mv_returns, ml_direct_returns, hrp_returns, rp_returns = [], [], [], []
    rolling_dates = []

    all_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)

    rebalance_map = {'M': 12, 'Q': 4, '6M': 2, 'Y': 1}
    rebalance_per_year = rebalance_map.get(rebalance_freq.upper(), 12)
    window_length = training_window_years * rebalance_per_year

    #weights_timeline = {
     #   'MV': [],
    #    'ML-MVO': [],
     #   'HRP': [],
     #   'Risk Parity': []
    #}
    dates = []

    for i in range(window_length, len(all_dates) - 1):
        print(f"[DEBUG] Starting window {i}: {all_dates[i - window_length]} to {all_dates[i]}")


        train_start = all_dates[i - window_length].strftime('%Y-%m-%d')
        train_end = all_dates[i].strftime('%Y-%m-%d')
        test_start = train_end
        test_end = all_dates[i + 1].strftime('%Y-%m-%d')

        try:
            train_data = download_stock_data(tickers, train_start, train_end)
            test_data = download_stock_data(tickers, test_start, test_end)
            test_returns = test_data['Adj Close'].pct_change().dropna()

            mv_weights = mean_variance_optimization(
                tickers, train_start, train_end, max_volatility,
                min_weight=min_weight, max_weight=max_weight
            )
            mv_weights = project_box_simplex(mv_weights, lo=min_weight, hi=max_weight)
            ret_mv = (test_returns.dot(mv_weights) + 1).prod() - 1
            mv_returns.append(ret_mv)

            try:
                investor_views = generate_walkforward_ml_views(
                tickers, train_start, train_end, target_mode='raw'
            )
            except Exception as e:
                print(f"[SKIP] Walk-forward failed from {train_start} to {train_end}: {e}")
                continue


            # Direct ML Predictions 
            try:
                ml_direct_vector = np.array([investor_views.get(t, 0.0) for t in tickers])
                ml_direct_mvo_weights = mean_variance_optimization(
                    tickers, train_start, train_end, max_volatility,
                    expected_returns=ml_direct_vector,
                    min_weight=min_weight, max_weight=max_weight
                )
                ml_direct_mvo_weights = project_box_simplex(ml_direct_mvo_weights, lo=min_weight, hi=max_weight)
                ret_ml_direct = (test_returns.dot(ml_direct_mvo_weights) + 1).prod() - 1
                ml_direct_returns.append(ret_ml_direct)
            except Exception as e:
                print(f"[SKIP] Direct ML failed from {train_start} to {train_end}: {e}")
                continue



            # HRP
            train_returns = train_data['Adj Close'].pct_change().dropna()
            cov = train_returns.cov()
            corr = train_returns.corr()
            hrp_weights_series = get_hrp_allocation(cov, corr)
            hrp_weights = hrp_weights_series.reindex(tickers).fillna(0).values
            hrp_weights = project_box_simplex(hrp_weights, lo=min_weight, hi=max_weight)
            ret_hrp = (test_returns.dot(hrp_weights) + 1).prod() - 1
            hrp_returns.append(ret_hrp)

            # Risk Parity
            rp_weights_series = get_risk_parity_weights(cov)
            rp_weights = rp_weights_series.reindex(tickers).fillna(0).values
            rp_weights = project_box_simplex(rp_weights, lo=min_weight, hi=max_weight)
            ret_rp = (test_returns.dot(rp_weights) + 1).prod() - 1
            rp_returns.append(ret_rp)

            rolling_dates.append(test_end)

        except Exception as e:
            print(f"[WARN] Skipping {test_start} to {test_end}: {e}")

    index = pd.to_datetime(rolling_dates)
    if len(index) == 0:
        raise ValueError("Backtest failed: No valid rolling windows. Check data availability or date range.")

    cumulative = lambda r: (1 + pd.Series(r, index=index)).cumprod()

    spy_data = download_stock_data(['SPY'], start_date, end_date)['Adj Close']
    spy_returns = spy_data.pct_change().dropna()
    spy_resampled = spy_returns.resample(rebalance_freq).apply(lambda x: (x + 1).prod() - 1)
    spy_cumulative = (1 + spy_resampled).cumprod()
    spy_cumulative = spy_cumulative.loc[spy_cumulative.index.intersection(index)]

    common_index = spy_cumulative.index
    unopt_cumulative_resampled = unopt_cumulative_resampled.loc[common_index]
    mv_series = cumulative(mv_returns).loc[common_index]
    ml_direct_series = cumulative(ml_direct_returns).loc[common_index]
    hrp_series = cumulative(hrp_returns).loc[common_index]
    rp_series = cumulative(rp_returns).loc[common_index]

    #weights_dfs = {k: pd.DataFrame(v) for k, v in weights_timeline.items()}

    return {
        'Unoptimized': unopt_cumulative_resampled,
        'MV': mv_series,
        'ML-MVO': ml_direct_series,  
        'HRP': hrp_series,
        'Risk Parity': rp_series
        #'SPY': spy_cumulative
    }



# --- paste in rebalancing.py (e.g., under existing functions) ---
import numpy as np
import pandas as pd

def run_monthly_weight_snapshots(
    portfolio_dict,
    start_date,
    end_date,
    training_window_years=5,
    rebalance_freq='M',
    max_volatility=0.225,
    min_weight=0.01,
    max_weight=0.25,
    target_mode='raw'  # 'raw' or 'excess_spy' to match your ML views
):
    """
    Compute ONLY the target weights at each rebalance date for:
      - MV (historical)
      - ML-MVO (direct predicted returns into MVO)
      - HRP
      - Risk Parity

    Returns:
        dict[str, pd.DataFrame]: keys in {'MV','ML-MVO','HRP','Risk Parity'}.
        Each value is a DataFrame with index=rebalancing dates, columns=tickers, values=weights.
    """
    # 1) Tickers from provided initial portfolio dict
    tickers, _ = calculate_weights(portfolio_dict)  
    assert len(tickers)*min_weight <= 1 <= len(tickers)*max_weight, (
        f"Infeasible bounds: n={len(tickers)}, min={min_weight}, max={max_weight}"
    )

    # 2) Build monthly/quarterly/... calendar
    all_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)

    # map freq -> periods per year to compute rolling window length
    rebalance_map = {'M': 12, 'Q': 4, '6M': 2, 'Y': 1}
    rebalance_per_year = rebalance_map.get(rebalance_freq.upper(), 12)
    window_length = training_window_years * rebalance_per_year

    # 3) Containers (lists of Series -> assembled to DataFrames at the end)
    timeline = {
        'MV': [],
        'ML-MVO': [],
        'HRP': [],
        'Risk Parity': []
    }

    # 4) Rolling loop
    for i in range(window_length, len(all_dates) - 1):
        train_start = all_dates[i - window_length].strftime('%Y-%m-%d')
        train_end   = all_dates[i].strftime('%Y-%m-%d')
        test_end    = all_dates[i + 1].strftime('%Y-%m-%d')  # label weights by *next* period end

        try:
            # pull data only for training window
            train_data = download_stock_data(tickers, train_start, train_end)
            train_adj  = train_data['Adj Close'].dropna(how='any')
            train_rets = train_adj.pct_change().dropna()

            # === MV ===
            try:
                mv_w = mean_variance_optimization(
                    tickers, train_start, train_end, max_volatility,
                    min_weight=min_weight, max_weight=max_weight
                )
                mv_w = project_box_simplex(mv_w, lo=min_weight, hi=max_weight)          
                timeline['MV'].append(pd.Series(mv_w, index=tickers, name=test_end))
            except Exception as e:
                print(f"[WARN MV weights] {train_start}->{train_end}: {e}")

            # === ML-MVO (Direct ML expected returns) ===
            try:
                investor_views = generate_walkforward_ml_views(
                    tickers, train_start, train_end, target_mode=target_mode
                )
                ml_vec = np.array([investor_views.get(t, 0.0) for t in tickers])
                mlmvo_w = mean_variance_optimization(
                    tickers, train_start, train_end, max_volatility,
                    expected_returns=ml_vec,
                    min_weight=min_weight, max_weight=max_weight
                )
                mlmvo_w = project_box_simplex(mlmvo_w, lo=min_weight, hi=max_weight)
                timeline['ML-MVO'].append(pd.Series(mlmvo_w, index=tickers, name=test_end))
            except Exception as e:
                print(f"[WARN ML-MVO weights] {train_start}->{train_end}: {e}")

            # === HRP ===
            try:
                cov = train_rets.cov()
                corr = train_rets.corr()
                hrp_w = get_hrp_allocation(cov, corr).reindex(tickers).fillna(0.0).values
                hrp_w = project_box_simplex(hrp_w, lo=min_weight, hi=max_weight)
                timeline['HRP'].append(pd.Series(hrp_w, index=tickers, name=test_end))
            except Exception as e:
                print(f"[WARN HRP weights] {train_start}->{train_end}: {e}")

            # === Risk Parity ===
            try:
                # reuse same cov from HRP block for efficiency
                rp_w = get_risk_parity_weights(cov).reindex(tickers).fillna(0.0).values
                rp_w = project_box_simplex(rp_w, lo=min_weight, hi=max_weight)
                timeline['Risk Parity'].append(pd.Series(rp_w, index=tickers, name=test_end))
            except Exception as e:
                print(f"[WARN RP weights] {train_start}->{train_end}: {e}")

        except Exception as e:
            print(f"[SKIP] {train_start}->{train_end}: {e}")
            continue

    # 5) Assemble clean DataFrames
    out = {}
    for k, series_list in timeline.items():
        if not series_list:
            continue
        dfk = pd.DataFrame(series_list)
        dfk.index = pd.to_datetime(dfk.index)
        out[k] = dfk.sort_index().reindex(columns=tickers).fillna(0.0)

    return out




import numpy as np
import pandas as pd

def backtest_strategies_with_weights(
    portfolio_dict,
    start_date,
    end_date,
    training_window_years=5,
    max_volatility=0.225,
    min_weight=0.01,
    max_weight=0.25,
    rebalance_freq='M',
    target_mode='raw',            # for ML views: 'raw' or 'excess_spy'
    include_spy=True,
    compute_turnover=True
):
    """
    Single-pass monthly/quarterly/... walk-forward backtest that returns:
      - monthly_results: dict of curves (pd.Series) + optional turnover series
      - weight_dfs: dict of per-strategy weight DataFrames (index=rebal dates)

    Relies on your existing helpers:
      calculate_weights, download_stock_data, mean_variance_optimization,
      project_box_simplex, get_hrp_allocation, get_risk_parity_weights,
      generate_walkforward_ml_views
    """

    # --- 0) Setup & sanity
    tickers, initial_weights = calculate_weights(portfolio_dict)
    n = len(tickers)
    assert n * min_weight <= 1.0 <= n * max_weight, (
        f"Infeasible bounds: n={n}, min={min_weight}, max={max_weight}"
    )

    # Rebalance calendar & window length
    all_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)
    if len(all_dates) < 3:
        raise ValueError("Insufficient rebalance dates in range.")

    rebalance_map = {'M': 12, 'Q': 4, '6M': 2, 'Y': 1}
    rebalance_per_year = rebalance_map.get(rebalance_freq.upper(), 12)
    window_length = training_window_years * rebalance_per_year

    # --- 1) Download once
    #   Core universe
    full_px = download_stock_data(tickers, start_date, end_date)['Adj Close'].dropna(how='all')
    if include_spy:
        spy_px = download_stock_data(['SPY'], start_date, end_date)['Adj Close'].dropna(how='all')
        spy_returns_daily = spy_px.pct_change().dropna()

    # Unoptimized path at daily freq -> resample & cumprod
    daily_returns_all = full_px.pct_change().dropna()
    unopt_daily = daily_returns_all.dot(initial_weights)
    unopt_cum_daily = (1.0 + unopt_daily).cumprod()
    # We’ll align to rolling index at the end, but keep a resampled series ready:
    unopt_resampled = unopt_cum_daily.resample(rebalance_freq).last()

    # --- 2) Containers
    # Period returns per window (we'll cumprod later)
    win_rets = {
        'MV': [],
        'ML-MVO': [],
        'HRP': [],
        'Risk Parity': []
    }
    rolling_dates = []  # label = end of test period

    # Weight timeline (for heatmaps) and optional turnover
    timeline = {k: [] for k in win_rets.keys()}
    if compute_turnover:
        turnover_lists = {k: [] for k in win_rets.keys()}
        prev_w = {k: None for k in win_rets.keys()}

    # --- 3) Walk-forward loop
    for i in range(window_length, len(all_dates) - 1):
        train_start = all_dates[i - window_length].strftime('%Y-%m-%d')
        train_end   = all_dates[i].strftime('%Y-%m-%d')
        test_start  = train_end
        test_end    = all_dates[i + 1].strftime('%Y-%m-%d')

        try:
            # Slice once from pre-fetched prices
            train_adj  = full_px.loc[train_start:train_end].dropna(how='any')
            if train_adj.empty:
                print(f"[SKIP] Empty training window {train_start}->{train_end}")
                continue

            train_rets = train_adj.pct_change().dropna()
            cov = train_rets.cov()
            corr = train_rets.corr()

            test_adj = full_px.loc[test_start:test_end]
            test_rets = test_adj.pct_change().dropna()
            if test_rets.empty:
                print(f"[SKIP] Empty test window {test_start}->{test_end}")
                continue

            # ==== MV ====
            try:
                mv_w = mean_variance_optimization(
                    tickers, train_start, train_end, max_volatility,
                    min_weight=min_weight, max_weight=max_weight
                )
                mv_w = project_box_simplex(mv_w, lo=min_weight, hi=max_weight)
                mv_ret = (1.0 + test_rets.dot(mv_w)).prod() - 1.0
                win_rets['MV'].append(mv_ret)
                timeline['MV'].append(pd.Series(mv_w, index=tickers, name=test_end))

                if compute_turnover:
                    if prev_w['MV'] is None:
                        turnover_lists['MV'].append(0.0)
                    else:
                        turnover_lists['MV'].append(0.5 * float(np.abs(mv_w - prev_w['MV']).sum()))
                    prev_w['MV'] = mv_w
            except Exception as e:
                print(f"[WARN MV] {train_start}->{train_end}: {e}")

            # ==== ML-MVO (Direct) ====
            try:
                views = generate_walkforward_ml_views(
                    tickers, train_start, train_end, target_mode=target_mode
                )
                ml_vec = np.array([views.get(t, 0.0) for t in tickers])
                ml_w = mean_variance_optimization(
                    tickers, train_start, train_end, max_volatility,
                    expected_returns=ml_vec,
                    min_weight=min_weight, max_weight=max_weight
                )
                ml_w = project_box_simplex(ml_w, lo=min_weight, hi=max_weight)
                ml_ret = (1.0 + test_rets.dot(ml_w)).prod() - 1.0
                win_rets['ML-MVO'].append(ml_ret)
                timeline['ML-MVO'].append(pd.Series(ml_w, index=tickers, name=test_end))

                if compute_turnover:
                    if prev_w['ML-MVO'] is None:
                        turnover_lists['ML-MVO'].append(0.0)
                    else:
                        turnover_lists['ML-MVO'].append(0.5 * float(np.abs(ml_w - prev_w['ML-MVO']).sum()))
                    prev_w['ML-MVO'] = ml_w
            except Exception as e:
                print(f"[WARN ML-MVO] {train_start}->{train_end}: {e}")

            # ==== HRP ====
            try:
                hrp_w = get_hrp_allocation(cov, corr).reindex(tickers).fillna(0.0).values
                hrp_w = project_box_simplex(hrp_w, lo=min_weight, hi=max_weight)
                hrp_ret = (1.0 + test_rets.dot(hrp_w)).prod() - 1.0
                win_rets['HRP'].append(hrp_ret)
                timeline['HRP'].append(pd.Series(hrp_w, index=tickers, name=test_end))

                if compute_turnover:
                    if prev_w['HRP'] is None:
                        turnover_lists['HRP'].append(0.0)
                    else:
                        turnover_lists['HRP'].append(0.5 * float(np.abs(hrp_w - prev_w['HRP']).sum()))
                    prev_w['HRP'] = hrp_w
            except Exception as e:
                print(f"[WARN HRP] {train_start}->{train_end}: {e}")

            # ==== Risk Parity ====
            try:
                rp_w = get_risk_parity_weights(cov).reindex(tickers).fillna(0.0).values
                rp_w = project_box_simplex(rp_w, lo=min_weight, hi=max_weight)
                rp_ret = (1.0 + test_rets.dot(rp_w)).prod() - 1.0
                win_rets['Risk Parity'].append(rp_ret)
                timeline['Risk Parity'].append(pd.Series(rp_w, index=tickers, name=test_end))

                if compute_turnover:
                    if prev_w['Risk Parity'] is None:
                        turnover_lists['Risk Parity'].append(0.0)
                    else:
                        turnover_lists['Risk Parity'].append(0.5 * float(np.abs(rp_w - prev_w['Risk Parity']).sum()))
                    prev_w['Risk Parity'] = rp_w
            except Exception as e:
                print(f"[WARN RP] {train_start}->{train_end}: {e}")

            # Only record the label if at least one strategy succeeded this window
            if any(k in win_rets and len(win_rets[k]) > 0 and len(timeline[k]) > 0 for k in win_rets.keys()):
                rolling_dates.append(test_end)

        except Exception as e:
            print(f"[SKIP] {train_start}->{train_end}: {e}")
            continue

    # --- 4) Pack to Series/DataFrames
    index = pd.to_datetime(rolling_dates)
    if len(index) == 0:
        raise ValueError("Backtest failed: No valid rolling windows. Check data availability or date range.")

    # helper to cumprod from period returns aligned with index
    def _cum(ret_list):
        return (1.0 + pd.Series(ret_list, index=index)).cumprod()

    # Curves we have
    out_curves = {}
    for k, lst in win_rets.items():
        if len(lst) == len(index):
            out_curves[k] = _cum(lst)

    # Unoptimized aligned to rolling calendar
    unopt_curve = unopt_resampled.loc[unopt_resampled.index.intersection(index)]
    # Reindex exactly to rolling dates using last available value up to that point
    unopt_curve = unopt_curve.reindex(index, method='pad')
    out_curves['Unoptimized'] = unopt_curve

    # Optional SPY curve aligned the same way
    if include_spy:
        spy_resampled = spy_returns_daily.resample(rebalance_freq).apply(lambda x: (1.0 + x).prod() - 1.0)
        spy_curve = (1.0 + spy_resampled).cumprod()
        spy_curve = spy_curve.loc[spy_curve.index.intersection(index)]
        spy_curve = spy_curve.reindex(index, method='pad').squeeze()
        out_curves['SPY'] = spy_curve

    # Weight DataFrames
    weight_dfs = {}
    for k, series_list in timeline.items():
        if series_list:
            dfk = pd.DataFrame(series_list)
            dfk.index = pd.to_datetime(dfk.index)
            weight_dfs[k] = dfk.sort_index().reindex(columns=tickers).fillna(0.0)

    # Optional turnover series, aligned to weight_dfs indices
    if compute_turnover:
        for k, lst in turnover_lists.items():
            if k in weight_dfs and len(lst) == len(weight_dfs[k].index):
                t_index = weight_dfs[k].index
                out_curves[f"Turnover:{k}"] = pd.Series(lst, index=t_index)

    return out_curves, weight_dfs
