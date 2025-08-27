import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, timedelta

# (better year math; safe fallback if not installed)
try:
    from dateutil.relativedelta import relativedelta
except Exception:
    relativedelta = None

# ===== Your modules =====
from machine_learning_strategies import (
    download_stock_data,
    generate_walkforward_ml_views  # dict or (views, confidences)
)
from mean_variance_optimization import mean_variance_optimization
from hrp import get_hrp_allocation
from rp import get_risk_parity_weights
from rebalancing import project_box_simplex  # STRICT min/max + sum=1

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("📊 Portfolio Optimizer — Weights by Strategy")
st.caption("Pick assets, choose a strategy, and get optimal weights + forward-looking metrics.")

# ---------------- Sidebar (Controls) ----------------
with st.sidebar:
    st.header("Controls")
    default_tickers = ['SPY','TLT','GLD','EFA','EEM','XLK','XLF','XLY','XLP','XLU']
    tickers_text = st.text_area("Tickers (comma-separated)", ",".join(default_tickers))
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

    years_back = st.number_input("Years of history", min_value=1, max_value=30, value=5, step=1)
    end_date = date.today()
    if relativedelta is not None:
        start_date = end_date - relativedelta(years=int(years_back))
    else:
        start_date = end_date - timedelta(days=int(years_back)*365)
    st.caption(f"Window: {start_date.isoformat()} → {end_date.isoformat()} (ends today)")

    strategy = st.selectbox(
        "Strategy",
        ["Equal Weight", "Mean-Variance (MVO)", "Hierarchical Risk Parity (HRP)", "Risk Parity (RP)", "ML-MVO (Direct views)"],
        index=1
    )
    max_vol = st.slider("Max annualized volatility (MVO/ML-MVO)", 0.05, 0.60, 0.25, 0.005)
    min_w = st.slider("Min weight per asset", 0.0, 0.25, 0.01, 0.01)
    max_w = st.slider("Max weight per asset", 0.10, 1.00, 0.35, 0.05)
    risk_free_ann = st.number_input("Risk-free rate (annual)", min_value=0.0, max_value=0.2, value=0.02, step=0.001)

    # 👉 Move the button into Controls
    run = st.button("Compute weights", type="primary", use_container_width=True)

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def dl_adj_close(tickers, start, end):
    """
    Uses your downloader; returns a clean Adj Close DataFrame (columns = tickers).
    """
    df = download_stock_data(tuple(tickers) if len(tickers) > 1 else tickers, str(start), str(end))
    if isinstance(df.columns, pd.MultiIndex):
        adj = df['Adj Close']
    else:
        if 'Adj Close' not in df.columns:
            raise ValueError("No 'Adj Close' in downloaded data.")
        adj = df[['Adj Close']]
        adj.columns = tickers
    return adj.dropna(how='all')

def eq_weight(tickers):
    return pd.Series(np.repeat(1.0/len(tickers), len(tickers)), index=tickers)

def mvo_weights(tickers, start, end, expected_returns=None):
    """
    expected_returns:
      - None → historical mean (annualized) inside mean_variance_optimization
      - If provided → pass MONTHLY LOG views; mean_variance_optimization() handles conversion.
    """
    w = mean_variance_optimization(
        tickers=tickers,
        start_date=str(start),
        end_date=str(end),
        max_volatility=float(max_vol),
        expected_returns=expected_returns,
        min_weight=float(min_w),
        max_weight=float(max_w),
        simulations=10000,
        seed=42
    )
    return pd.Series(w, index=tickers)

def hrp_weights_from_prices(adj_close: pd.DataFrame) -> pd.Series:
    rets = adj_close.pct_change().dropna()
    cov = rets.cov()
    corr = rets.corr()
    w = get_hrp_allocation(cov, corr)
    return w.reindex(adj_close.columns).fillna(0.0)

def rp_weights_from_prices(adj_close: pd.DataFrame) -> pd.Series:
    rets = adj_close.pct_change().dropna()
    cov = rets.cov()
    w = get_risk_parity_weights(cov)
    return (w / w.sum()).reindex(adj_close.columns).fillna(0.0)

def as_series(w, idx):
    """Coerce any weights-like object into a pandas Series indexed by tickers."""
    if isinstance(w, pd.Series):
        return w.reindex(idx).astype(float)
    return pd.Series(np.asarray(w, dtype=float).ravel(), index=idx)

# ---- Ex-ante metrics helpers ----
def ex_ante_vol_annual(weights: pd.Series, adj_close: pd.DataFrame) -> float:
    """Annualized portfolio volatility from daily covariance (√252)."""
    w = pd.Series(weights, index=adj_close.columns).fillna(0.0).values.astype(float)
    daily = adj_close.pct_change().dropna(how="all")
    if daily.shape[0] < 2:
        return float("nan")
    cov_d = daily.cov().values
    var_ann = float(w @ cov_d @ w) * 252.0
    return float(np.sqrt(max(var_ann, 0.0)))

def expected_return_mvo_monthly(weights: pd.Series, adj_close: pd.DataFrame) -> float:
    """Historical mean *monthly* arithmetic return over the selected window (for non-ML strategies)."""
    monthly = adj_close.resample("M").last().pct_change().dropna(how="all")
    if monthly.empty:
        return float("nan")
    mu_m = monthly.mean().astype(float)
    w = pd.Series(weights).reindex(mu_m.index).fillna(0.0).astype(float)
    return float(w.values @ mu_m.values)  # decimal monthly

def expected_return_ml_monthly(weights: pd.Series,
                               ml_views: dict | pd.Series | np.ndarray,
                               tickers: list[str]) -> float:
    """ML views are next-1M *log* returns. Convert to arithmetic: exp(mu_log)−1, then weight."""
    if isinstance(ml_views, dict):
        v = pd.Series({t: ml_views.get(t, 0.0) for t in tickers}, index=tickers, dtype=float)
    elif isinstance(ml_views, pd.Series):
        v = ml_views.reindex(tickers).fillna(0.0).astype(float)
    else:
        v = pd.Series(np.asarray(ml_views, float).ravel(), index=tickers)
    w = pd.Series(weights).reindex(tickers).fillna(0.0).astype(float)
    r_m = np.expm1(v.values)  # arithmetic monthly
    return float(np.dot(w.values, r_m))  # decimal monthly

def to_annual_from_monthly(mu_m: float) -> float:
    """Compound a monthly expected return to annual: (1+μm)^12 − 1."""
    if pd.isna(mu_m):
        return float("nan")
    return float((1.0 + mu_m)**12 - 1.0)

# ---------------- Action ----------------
if run:  # ✅ use the sidebar button
    if len(tickers) < 2:
        st.error("Please enter at least two tickers.")
        st.stop()

    try:
        adj_close = dl_adj_close(tickers, start_date, end_date)
        adj_close = adj_close.dropna(axis=1, how='all')
        missing = [t for t in tickers if t not in adj_close.columns]
        if missing:
            st.warning(f"Dropped tickers with no data: {', '.join(missing)}")
            tickers = [t for t in tickers if t in adj_close.columns]
        if len(tickers) < 2:
            st.error("Not enough valid series after dropping missing tickers.")
            st.stop()

        # ---- Feasibility check for bounds BEFORE computing weights ----
        n = len(tickers)
        if n * float(min_w) > 1.0 or n * float(max_w) < 1.0:
            st.error(
                f"Infeasible bounds for n={n}: n*min={n*float(min_w):.2f}, n*max={n*float(max_w):.2f}. "
                "Loosen min/max or change the ticker list."
            )
            st.stop()

        # ---- Compute raw weights per strategy ----
        ml_views_only = None  # keep for ML-MVO metrics later
        if strategy == "Equal Weight":
            weights = eq_weight(tickers)

        elif strategy == "Mean-Variance (MVO)":
            try:
                weights = mvo_weights(tickers, start_date, end_date, expected_returns=None)
            except Exception as e:
                st.error("No feasible MVO under current settings (vol cap / date range / bounds). Try loosening them.")
                st.exception(e)
                st.stop()

        elif strategy == "Hierarchical Risk Parity (HRP)":
            weights = hrp_weights_from_prices(adj_close)

        elif strategy == "Risk Parity (RP)":
            weights = rp_weights_from_prices(adj_close)

        elif strategy == "ML-MVO (Direct views)":
            with st.spinner("Generating pooled ML views (RS + WF)…"):
                ml_views = generate_walkforward_ml_views(
                    tickers=tickers,
                    start_date=str(start_date),
                    end_date=str(end_date),
                    target_horizon=20,
                    n_splits=5,
                    n_iter_rs=20,
                    random_state=42,
                    refit_on_full_data=True,
                    target_mode='excess_spy'
                )
            if isinstance(ml_views, tuple) and len(ml_views) >= 1:
                ml_views_only = ml_views[0]
            else:
                ml_views_only = ml_views

            expected_monthly_log = np.array([ml_views_only.get(t, 0.0) for t in tickers], dtype=float)
            try:
                weights = mvo_weights(tickers, start_date, end_date, expected_returns=expected_monthly_log)
            except Exception as e:
                st.error("No feasible ML-MVO under current settings (vol cap / date range / bounds). Try loosening them.")
                st.exception(e)
                st.stop()
        else:
            st.error("Unknown strategy.")
            st.stop()

        # ---- Coerce → enforce strict [min_w, max_w] + sum=1 → keep Series ----
        weights = as_series(weights, tickers)
        weights = project_box_simplex(weights, float(min_w), float(max_w))
        weights = as_series(weights, tickers)

        # ---------- Ex-ante Metrics Panel (4 tiles) ----------
        c_top1, c_top2, c_top3, c_top4 = st.columns([1, 1, 1, 1])

        # 1) Volatility (annualized) — for all strategies
        vol_ann = ex_ante_vol_annual(weights, adj_close)  # decimal annual
        c_top1.metric("Ex-ante Volatility (annualized)", f"{vol_ann*100:.2f}%")

        # 2) Expected monthly return source
        if strategy == "ML-MVO (Direct views)":
            er_m = expected_return_ml_monthly(weights, ml_views_only, tickers)
            er_caption = "ML next-1M (log→arith)."
        else:
            er_m = expected_return_mvo_monthly(weights, adj_close)
            er_caption = "Historical mean monthly (selected window)."

        c_top2.metric("Expected Return (next month)", f"{er_m*100:.2f}%" if pd.notna(er_m) else "—")

        # 3) Expected Sharpe (annualized)
        er_ann = to_annual_from_monthly(er_m)  # decimal annual
        exp_sharpe = (er_ann - float(risk_free_ann)) / vol_ann if (pd.notna(er_ann) and pd.notna(vol_ann) and vol_ann > 0) else np.nan
        c_top3.metric("Expected Sharpe (annualized)", f"{exp_sharpe:.2f}" if pd.notna(exp_sharpe) else "—")

        # 4) Expected Sortino (annualized) — DAILY downside risk to match Sharpe basis
        daily = adj_close.pct_change().dropna(how="any")
        if daily.empty:
            exp_sortino = np.nan
        else:
            w_vec = weights.reindex(daily.columns).fillna(0.0).values
            port_d = pd.Series(daily.values @ w_vec, index=daily.index)  # daily portfolio returns
            mar_d = float(risk_free_ann) / 252.0
            downside_d = np.minimum(port_d - mar_d, 0.0)
            dd_ann = downside_d.std(ddof=1) * np.sqrt(252)  # annualized downside deviation
            exp_sortino = ((er_ann - float(risk_free_ann)) / dd_ann) if (dd_ann and not np.isnan(dd_ann)) else np.nan

        c_top4.metric("Expected Sortino (annualized)", f"{exp_sortino:.2f}" if pd.notna(exp_sortino) else "—")

        st.caption(
            f"ER source: {er_caption} Risk basis: daily covariance (Sharpe) and daily downside deviation (Sortino), both annualized. "
            "Ex-ante estimates, not guarantees."
        )
        # -------------------------------------------

        # ---- Finalize & display weights (do NOT re-normalize again) ----
        weights = weights.reindex(tickers).fillna(0.0)

        st.subheader(f"Optimal Weights — {strategy}")
        c1, c2 = st.columns([2, 1])
        with c1:
            chart_df = pd.DataFrame({"Ticker": weights.index, "Weight": weights.values}).sort_values("Weight", ascending=False)
            st.bar_chart(chart_df.set_index("Ticker"))
        with c2:
            st.dataframe(pd.DataFrame({"Weight": weights}).style.format("{:.2%}"), use_container_width=True)

        st.success("Done.")

    except Exception as e:
        st.exception(e)
