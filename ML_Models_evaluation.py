"""
Machine Learning Return Forecasting — Evaluation Script (Appendix-Ready)
=======================================================================

Output
------
• CSVs and PNGs under outputs/ml_eval_<timestamp>/reports/
• Combined table: WF_RS_median_and_pooled_metrics.csv
"""
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from feature_engine.outliers import Winsorizer
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from scipy.stats import randint, uniform, spearmanr
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold


try:
    import yfinance as yf
except Exception as e:
    raise SystemExit(
        "yfinance is required. Install with:\n"
        "  pip install yfinance numpy pandas scikit-learn scipy matplotlib"
    )

# -----------------------
# Configuration
# -----------------------
TICKERS = [
    'SPY', 'TLT', 'GLD', 'VNQ', 'EFA', 'EEM', 'XLF', 'XLK',
    'XLP', 'XLY', 'XLU', 'XLE', 'TIP', 'HYG', 'LQD', 'DBC'
]
START = '2010-01-01'
END   = '2025-06-30'

N_SPLITS_WF   = 5
N_ITER_RS     = 20
INNER_CV_RS   = 3
RANDOM_SEED   = 42
np.random.seed(RANDOM_SEED)

STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTDIR = Path("outputs") / f"ml_eval_{STAMP}"
OUTDIR.mkdir(parents=True, exist_ok=True)
REPORTS = OUTDIR / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
# -----------------------
# Feature importance helpers (minimal, fold-aware)
# -----------------------
def _extract_feature_names_after_var(fitted_pipeline, original_cols):
    # Return feature names retained after the fitted VarianceThreshold in the pipeline.
    var = fitted_pipeline.named_steps.get('var', None)
    if var is None:
        return list(original_cols)
    mask = var.get_support()
    return list(pd.Index(original_cols)[mask])

def _feature_importance_from_fitted_pipeline(fitted_pipeline, kept_cols, model_name):
    # Given a fitted pipeline (imp -> var -> model), return a Series of importances aligned to kept_cols.
    mdl = fitted_pipeline.named_steps['model']
    if hasattr(mdl, 'feature_importances_'):
        imp = np.asarray(mdl.feature_importances_, dtype=float)
        return pd.Series(imp, index=kept_cols, name=f"{model_name}_importance")
    elif hasattr(mdl, 'coef_'):
        coef = np.asarray(mdl.coef_, dtype=float).ravel()
        s_signed = pd.Series(coef, index=kept_cols, name=f"{model_name}_coef_signed")
        s_abs = s_signed.abs()
        s_abs.name = f"{model_name}_coef_abs"
        return s_abs
    else:
        return None

def _group_ticker_dummies(importance_series: pd.Series, prefix='tk_'):
    # Collapse all ticker one-hot columns into a single 'Ticker_dummies' entry.
    s = importance_series.copy()
    idx_series = pd.Series(s.index.astype(str), index=s.index)
    mask = idx_series.str.startswith(prefix)
    grouped_val = float(s[mask].sum()) if mask.any() else 0.0
    s = s[~mask]
    if grouped_val > 0:
        s.loc['Ticker_dummies'] = grouped_val
    return s.sort_values(ascending=False)

def _save_importance_artifacts(imp_mean: pd.Series, imp_std: pd.Series, model_name: str, reports_dir: Path):
    # Save CSV and a horizontal barplot of the top 20 features.
    df = pd.DataFrame({'importance_mean': imp_mean, 'importance_std': imp_std})
    df.sort_values('importance_mean', ascending=False).to_csv(
        reports_dir / f"{model_name.replace(' ', '_')}_RS_WF_feature_importance.csv"
    )
    top = df.sort_values('importance_mean', ascending=False).head(20)
    plt.figure(figsize=(8, 6))
    plt.barh(top.index[::-1], top['importance_mean'].values[::-1])
    plt.xlabel('Importance (mean across folds)')
    plt.title(f"Feature Importance — {model_name} (RS + WF)")
    plt.tight_layout()
    plt.savefig(reports_dir / f"{model_name.replace(' ', '_')}_RS_WF_feature_importance_top20.png", dpi=300)
    plt.close()


# -----------------------
# Data utilities
# -----------------------
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()

def _to_monthly_close(df: pd.DataFrame) -> pd.DataFrame:
    m = pd.DataFrame()
    m['Adj Close'] = df['Adj Close'].resample('M').last()
    return m.dropna()

def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data for {ticker}")
    df = _ensure_datetime_index(df)
    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']
    return df[['Open','High','Low','Close','Adj Close','Volume']].dropna()

def fetch_macro_monthly(start: str, end: str) -> pd.DataFrame:
    macro_tickers = {
        'VIX': '^VIX', 'TNX': '^TNX', 'IRX': '^IRX',
        'WTI': 'CL=F', 'GOLD': 'GC=F', 'USD': 'UUP',
        'ACWX': 'ACWX', 'HYG': 'HYG', 'LQD': 'LQD'
    }
    raw = {}
    for k, t in macro_tickers.items():
        d = download_stock_data(t, start, end)
        d_m = _to_monthly_close(d)
        raw[k] = d_m.rename(columns={'Adj Close': k})
    macro = None
    for k, dfk in raw.items():
        macro = dfk if macro is None else macro.join(dfk, how='outer')
    macro = macro.dropna()
    feats = pd.DataFrame(index=macro.index)
    feats['TNX'] = macro['TNX']; feats['IRX'] = macro['IRX']
    feats['TERM_SPREAD'] = feats['TNX'] - feats['IRX']
    feats['dTNX'] = feats['TNX'].diff(); feats['dIRX'] = feats['IRX'].diff()
    feats['dTERM'] = feats['TERM_SPREAD'].diff()
    feats['VIX'] = macro['VIX']; feats['dVIX'] = macro['VIX'].diff()
    for k in ['WTI','GOLD','USD','ACWX','HYG','LQD']:
        feats[f'{k}_ret1'] = np.log(macro[k]).diff()
    feats['CREDIT_ret_spread'] = feats['HYG_ret1'] - feats['LQD_ret1']
    feats['CREDIT_ratio_chg'] = np.log(macro['HYG'] / macro['LQD']).diff()
    feats = feats.shift(1).dropna()  # lag 1M
    return feats

def build_panel_with_macros(tickers, start, end, target_horizon_months=1,
                            target_mode='raw', use_smooth_3m=False):
    macro = fetch_macro_monthly(start, end)
    rows = []
    for t in tickers:
        stock_d = download_stock_data(t, start, end)
        stock_m = _to_monthly_close(stock_d).rename(columns={'Adj Close':'PX'})
        df = stock_m.join(macro, how='outer')
        px = df['PX']
        logp = np.log(px)
        ret1 = logp.diff()
        mom12 = ret1.rolling(12).sum() - ret1
        mom6  = ret1.rolling(6).sum() - ret1
        rev1  = -ret1
        vol3  = ret1.rolling(3).std()
        vol6  = ret1.rolling(6).std()
        vol12 = ret1.rolling(12).std()
        roll_max12 = px.rolling(12).max()
        ddown12 = (px / roll_max12) - 1.0
        sma3  = px.rolling(3).mean();  sma6 = px.rolling(6).mean();  sma12 = px.rolling(12).mean()
        sma3r = px / sma3; sma6r = px / sma6; sma12r = px / sma12
        feat = pd.DataFrame({
            'mom_12_1': mom12, 'mom_6_1': mom6, 'rev_1m': rev1,
            'vol_3m': vol3, 'vol_6m': vol6, 'vol_12m': vol12,
            'ddown_12m': ddown12, 'sma3_ratio': sma3r,
            'sma6_ratio': sma6r, 'sma12_ratio': sma12r
        }, index=df.index).join(macro, how='left')
        fwd = ret1.shift(-target_horizon_months)
        if use_smooth_3m:
            fwd = ret1.shift(-1).rolling(3).mean()
        target = fwd  # raw
        feat['target'] = target
        feat['Ticker'] = t
        rows.append(feat)
    panel = pd.concat(rows).dropna()
    tk = pd.get_dummies(panel['Ticker'], prefix='tk', drop_first=True)
    X = pd.concat([panel.drop(columns=['target','Ticker']), tk], axis=1)
    y = panel['target']
    ql, qh = y.quantile([0.01, 0.99])
    y = y.clip(ql, qh)
    # Do NOT fit imputers on the full period. Return raw X with NaNs.
    mi = pd.MultiIndex.from_arrays([X.index, panel['Ticker']], names=['Date','Ticker'])
    X.index = mi
    y.index = mi
    return X, y, X.columns.tolist()

# -----------------------
# Metrics
# -----------------------
def calc_hit_rate(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))

def calc_ic(y_true, y_pred) -> float:
    try:
        ic = spearmanr(y_true, y_pred, nan_policy='omit')[0]
    except Exception:
        ic = np.nan
    return float(ic)

# -----------------------
# RS helper & WF eval
# -----------------------
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

def fit_with_random_search(model, param_dist, X_tr, y_tr,
                           n_iter=N_ITER_RS, inner_cv=INNER_CV_RS, random_state=RANDOM_SEED):
    """
    Fold-local pipeline so preprocessing fits ONLY on training data.
    """
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='mean')),
        ('var', VarianceThreshold(threshold=0.0)),
        ('model', model),
    ])

    if not param_dist:
        pipe.fit(X_tr, y_tr)
        return pipe

    param_prefixed = {f"model__{k}": v for k, v in param_dist.items()}
    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_prefixed,
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=inner_cv),
        scoring='r2',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    rs.fit(X_tr, y_tr)
    return rs.best_estimator_


def walkforward_per_model_RS(model_spec, X, y, n_splits=N_SPLITS_WF, model_name="Model"):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    y_true_all, y_pred_all, idx_all = [], [], []
    imp_list = []  # per-fold importances
    for fold, (tr, te) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        # Purge the first test month to avoid train/test edge overlap
        first_test_date = y_te.index.get_level_values('Date').min()
        mask = y_te.index.get_level_values('Date') > first_test_date
        X_te, y_te = X_te.loc[mask], y_te.loc[mask]
        if y_te.empty:
            continue
        best_model = fit_with_random_search(model_spec['base'], model_spec['params'], X_tr, y_tr)
        # importance per fold
        try:
            kept_cols = _extract_feature_names_after_var(best_model, X_tr.columns)
            imp_series = _feature_importance_from_fitted_pipeline(best_model, kept_cols, model_name)
            if isinstance(imp_series, pd.Series):
                imp_list.append(imp_series)
        except Exception as _e:
            pass
        y_hat = best_model.predict(X_te)
        y_true_all.append(y_te.values)
        y_pred_all.append(y_hat)
        idx_all.append(y_te.index)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    idx_concat = idx_all[0]
    for k in range(1, len(idx_all)):
        idx_concat = idx_concat.append(idx_all[k])
    true_series = pd.Series(y_true_all, index=idx_concat, name='True')
    pred_series = pd.Series(y_pred_all, index=idx_concat, name='Pred')
    rows = []
    tickers = sorted(set(true_series.index.get_level_values('Ticker')))
    for tk in tickers:
        yt = true_series.xs(tk, level='Ticker')
        yp_ = pred_series.xs(tk, level='Ticker')
        common = yt.index.intersection(yp_.index)
        yt, yp_ = yt.loc[common], yp_.loc[common]
        if len(common) < 3:
            r2, mae, hit, ic = np.nan, np.nan, np.nan, np.nan
        else:
            r2  = r2_score(yt, yp_)
            mae = mean_absolute_error(yt, yp_)
            hit = calc_hit_rate(yt, yp_)
            ic  = calc_ic(yt, yp_)
        rows.append({'Model': model_name, 'Ticker': tk, 'R2': r2, 'MAE': mae, 'HitRate': hit, 'IC': ic})
    per_ticker_df = pd.DataFrame(rows).sort_values(['Model','Ticker'])
    r2_pooled  = r2_score(true_series, pred_series)
    mae_pooled = mean_absolute_error(true_series, pred_series)
    hit_pooled = calc_hit_rate(true_series, pred_series)
    ic_pooled  = calc_ic(true_series, pred_series)
    # aggregate importances across folds
    imp_mean = imp_std = None
    if len(imp_list) > 0:
        imp_df = pd.concat(imp_list, axis=1)
        imp_df.columns = [f'fold_{i+1}' for i in range(imp_df.shape[1])]
        imp_mean = imp_df.mean(axis=1).sort_values(ascending=False)
        imp_std  = imp_df.std(axis=1).reindex(imp_mean.index)
    return per_ticker_df, true_series, pred_series, r2_pooled, mae_pooled, hit_pooled, ic_pooled, (imp_mean, imp_std)

# -----------------------
# Model spaces
# -----------------------
MODELS = {
    'Linear Regression': {
        'base': LinearRegression(),
        'params': {}
    },
    'Random Forest': {
        'base': RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        'params': {
            'n_estimators': randint(50, 400),
            'max_depth': randint(2, 12),
            'min_samples_split': randint(2, 12),
            'min_samples_leaf': randint(1, 8)
        }
    },
    'Gradient Boosting': {
        'base': GradientBoostingRegressor(random_state=RANDOM_SEED),
        'params': {
            'n_estimators': randint(50, 500),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(2, 6),
            'subsample': uniform(0.6, 0.4),
            'min_samples_leaf': randint(1, 8),
            'max_features': ['sqrt', 'log2', None]
        }
    },
}

# -----------------------
# Main
# -----------------------
def main():
    print("Building monthly panel ...", flush=True)
    Xp, yp, _ = build_panel_with_macros(
        TICKERS, START, END,
        target_horizon_months=1,
        target_mode='raw',
        use_smooth_3m=False
    )

    results_tables = []
    pooled_summary = []
    all_preds = {}

    for name, spec in MODELS.items():
        # Walk-forward with RS only
        tbl, y_true_s, y_pred_s, r2_pool, mae_pool, hit_pool, ic_pool, (imp_mean, imp_std) = walkforward_per_model_RS(
            spec, Xp, yp, n_splits=N_SPLITS_WF, model_name=name
        )
        results_tables.append(tbl)
        pooled_summary.append({'Model': name, 'R2_pooled': r2_pool, 'MAE_pooled': mae_pool,
                               'HitRate_pooled': hit_pool, 'IC_pooled': ic_pool})
        all_preds[name] = {'true': y_true_s, 'pred': y_pred_s}

        # Save per-model per-ticker metrics
        csv_path = REPORTS / f"{name.replace(' ', '_')}_RS_WF_metrics_per_ticker.csv"
        tbl.to_csv(csv_path, index=False)

        # Scatter (pooled)
        fig_path = REPORTS / f"{name.replace(' ', '_')}_RS_WF_actual_vs_pred.png"
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_s, y_pred_s, alpha=0.35, s=10)
        plt.axhline(0, color='gray', lw=1); plt.axvline(0, color='gray', lw=1)
        lim = max(abs(y_true_s.min()), abs(y_true_s.max()), abs(y_pred_s.min()), abs(y_pred_s.max()))
        lim = float(lim) if np.isfinite(lim) else 0.05
        plt.plot([-lim, lim], [-lim, lim], linestyle='--', linewidth=1)
        plt.xlim(-lim, lim); plt.ylim(-lim, lim)
        plt.xlabel("Actual 1M Return"); plt.ylabel("Predicted 1M Return")
        plt.title(f"Actual vs Predicted — {name} (RS + WF)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(fig_path, dpi=300); plt.close()
        # Save feature importance artifacts
        if imp_mean is not None:
            _save_importance_artifacts(imp_mean, imp_std, name, REPORTS)
            # Grouped ticker dummies
            imp_mean_grp = _group_ticker_dummies(imp_mean, prefix='tk_')
            imp_std_grp  = imp_std.reindex(imp_mean_grp.index) if imp_std is not None else None
            _save_importance_artifacts(imp_mean_grp, imp_std_grp, f"{name}_Grouped", REPORTS)


    # Ensemble = mean(RF, GB)
    print("Building Ensemble (mean of RF, GB) ...", flush=True)
    if not {'Random Forest','Gradient Boosting'}.issubset(set(all_preds.keys())):
        raise RuntimeError("RF and GB predictions are required to build the ensemble.")
    true_ens = all_preds['Random Forest']['true'].copy()
    pred_cols = [
        all_preds['Random Forest']['pred'].reindex(true_ens.index).values,
        all_preds['Gradient Boosting']['pred'].reindex(true_ens.index).values
    ]
    pred_stack = np.column_stack(pred_cols)
    pred_mean = pred_stack.mean(axis=1)
    pred_ens = pd.Series(pred_mean, index=true_ens.index, name='Pred')

    rows_ens = []
    for tk in sorted(set(true_ens.index.get_level_values('Ticker'))):
        yt = true_ens.xs(tk, level='Ticker')
        yp_ = pred_ens.xs(tk, level='Ticker')
        common = yt.index.intersection(yp_.index)
        yt, yp_ = yt.loc[common], yp_.loc[common]
        if len(common) < 3:
            r2, mae, hit, ic = np.nan, np.nan, np.nan, np.nan
        else:
            r2  = r2_score(yt, yp_)
            mae = mean_absolute_error(yt, yp_)
            hit = calc_hit_rate(yt, yp_)
            ic  = calc_ic(yt, yp_)
        rows_ens.append({'Model': 'Ensemble', 'Ticker': tk, 'R2': r2, 'MAE': mae, 'HitRate': hit, 'IC': ic})
    ens_table = pd.DataFrame(rows_ens).sort_values(['Model','Ticker'])
    ens_table.to_csv(REPORTS / "Ensemble_RS_WF_metrics_per_ticker.csv", index=False)

    r2_pool_ens  = r2_score(true_ens, pred_ens)
    mae_pool_ens = mean_absolute_error(true_ens, pred_ens)
    hit_pool_ens = calc_hit_rate(true_ens, pred_ens)
    ic_pool_ens  = calc_ic(true_ens, pred_ens)
    pooled_summary.append({'Model': 'Ensemble', 'R2_pooled': r2_pool_ens, 'MAE_pooled': mae_pool_ens,
                           'HitRate_pooled': hit_pool_ens, 'IC_pooled': ic_pool_ens})

    # Ensemble scatter
    fig_path_ens = REPORTS / "Ensemble_RS_WF_actual_vs_pred.png"
    plt.figure(figsize=(6, 6))
    plt.scatter(true_ens, pred_ens, alpha=0.35, s=10)
    plt.axhline(0, color='gray', lw=1); plt.axvline(0, color='gray', lw=1)
    lim = max(abs(true_ens.min()), abs(true_ens.max()), abs(pred_ens.min()), abs(pred_ens.max()))
    lim = float(lim) if np.isfinite(lim) else 0.05
    plt.plot([-lim, lim], [-lim, lim], linestyle='--', linewidth=1)
    plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.xlabel("Actual 1M Return"); plt.ylabel("Predicted 1M Return")
    plt.title("Actual vs Predicted — Ensemble (RS + WF)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_path_ens, dpi=300); plt.close()

    # Final tables
    per_ticker_all = pd.concat(results_tables + [ens_table], ignore_index=True)
    summary_all = pd.DataFrame(pooled_summary).sort_values('Model')
    per_ticker_all.to_csv(REPORTS / "WF_RS_metrics_per_ticker_ALL_MODELS.csv", index=False)
    summary_all.to_csv(REPORTS / "WF_RS_pooled_metrics_ALL_MODELS.csv", index=False)

    median_metrics = (
        per_ticker_all
        .groupby('Model')[['R2', 'MAE', 'HitRate', 'IC']]
        .median()
        .rename(columns={'R2':'Median_R2','MAE':'Median_MAE','HitRate':'Median_HitRate','IC':'Median_IC'})
    )
    pooled_metrics = summary_all.set_index('Model')[['R2_pooled','MAE_pooled','HitRate_pooled','IC_pooled']]
    combined_metrics = median_metrics.join(pooled_metrics)[
        ['Median_R2','Median_MAE','Median_HitRate','Median_IC','R2_pooled','MAE_pooled','HitRate_pooled','IC_pooled']
    ]
    combined_path = REPORTS / "WF_RS_median_and_pooled_metrics.csv"
    combined_metrics.to_csv(combined_path)

    print("\n=== Summary (median per ticker & pooled) ===", flush=True)
    print(combined_metrics.round(6).to_string())
    print(f"\nAll outputs → {REPORTS}")

if __name__ == "__main__":
    main()


# =========================
# === LINE PLOT HELPERS ===
# =========================
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# If yfinance isn't already imported in your file:
try:
    import yfinance as yf
except Exception as _e:
    raise SystemExit("yfinance is required (pip install yfinance)")

OUT_DIR = Path("outputs_lineplots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _plot_spy_monthly_line(start, end):
    """Saves SPY monthly Adj Close line to OUT_DIR/SPY_monthly_line.png"""
    df = yf.download("SPY", start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError("No SPY data downloaded.")
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    monthly = df["Adj Close"].resample("M").last().dropna()
    plt.figure(figsize=(10, 5))
    plt.plot(monthly.index, monthly.values, linewidth=1.5)
    plt.title("SPY — Monthly Adj Close")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "SPY_monthly_line.png", dpi=200)
    plt.close()

def _per_ticker_r2(y_true_s: pd.Series, y_pred_s: pd.Series) -> pd.DataFrame:
    """
    Computes per-ticker R² and returns a DataFrame sorted desc by R².
    Assumes y_* are MultiIndex Series with level 'Ticker'.
    """
    from sklearn.metrics import r2_score
    rows = []
    tickers = sorted(set(y_true_s.index.get_level_values("Ticker")))
    for tk in tickers:
        yt = y_true_s.xs(tk, level="Ticker")
        yp = y_pred_s.xs(tk, level="Ticker")
        common = yt.index.intersection(yp.index)
        if len(common) < 3:
            r2 = np.nan
        else:
            r2 = r2_score(yt.loc[common], yp.loc[common])
        rows.append({"Ticker": tk, "R2": r2})
    return pd.DataFrame(rows).sort_values("R2", ascending=False)

def _plot_actual_vs_pred_lines(model_name: str, y_true_s: pd.Series, y_pred_s: pd.Series, ticker: str):
    """Line plot of Actual vs Predicted 1M returns for a single ticker."""
    yt = y_true_s.xs(ticker, level="Ticker")
    yp = y_pred_s.xs(ticker, level="Ticker")
    common = yt.index.intersection(yp.index)
    yt, yp = yt.loc[common], yp.loc[common]

    plt.figure(figsize=(10, 5))
    plt.plot(yt.index, yt.values, label="Actual", linewidth=1.5)
    plt.plot(yp.index, yp.values, label="Predicted", linewidth=1.5)
    plt.title(f"{model_name} — Actual vs Predicted (Best R²) — {ticker}")
    plt.xlabel("Date")
    plt.ylabel("1M Return (log or simple, matching your target)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fname = f"Line_Actual_vs_Pred_{model_name.replace(' ','_')}_{ticker}.png"
    plt.savefig(OUT_DIR / fname, dpi=200)
    plt.close()

def make_spy_and_best_r2_plots():
    """
    1) Build the same panel you use elsewhere (identical config).
    2) Save SPY monthly line.
    3) For RF, GB: compute RS+WF predictions, pick best-R² ticker, plot lines.
    4) Build Ensemble = mean(RF, GB) predictions; pick best-R² ticker; plot lines.
    """
    # --- 1) Build panel exactly like your evaluation code ---
    Xp, yp, _ = build_panel_with_macros(
        TICKERS, START, END,
        target_horizon_months=1,
        target_mode='raw',         # <- change to 'excess_spy' if that's what you evaluate elsewhere
        use_smooth_3m=False
    )

    # --- 2) SPY line ---
    _plot_spy_monthly_line(START, END)

    # --- 3) RF & GB RS+WF predictions ---
    model_outputs = {}
    for name in ["Random Forest", "Gradient Boosting"]:
        spec = MODELS[name]  # expects your dict of model specs
        # walkforward_per_model_RS should return (per_ticker_df, y_true_s, y_pred_s, ...)
        per_df, y_true_s, y_pred_s, *_ = walkforward_per_model_RS(
            spec, Xp, yp, n_splits=5, model_name=name
        )
        model_outputs[name] = {"per_df": per_df, "true": y_true_s, "pred": y_pred_s}

        # pick best R² ticker
        ranking = _per_ticker_r2(y_true_s, y_pred_s)
        best_ticker = ranking.iloc[0]["Ticker"]
        _plot_actual_vs_pred_lines(name, y_true_s, y_pred_s, best_ticker)

    # --- 4) Ensemble = mean of RF & GB predictions (aligned on the same MultiIndex) ---
    y_true_ens = model_outputs["Random Forest"]["true"]
    yp_rf = model_outputs["Random Forest"]["pred"].reindex(y_true_ens.index)
    yp_gb = model_outputs["Gradient Boosting"]["pred"].reindex(y_true_ens.index)

    # Handle any missing values before averaging
    arr = np.column_stack([
        yp_rf.values.astype(float),
        yp_gb.values.astype(float)
    ])
    pred_mean = np.nanmean(arr, axis=1)

    y_pred_ens = pd.Series(pred_mean, index=y_true_ens.index, name="Pred")

    ens_rank = _per_ticker_r2(y_true_ens, y_pred_ens)
    best_ens_ticker = ens_rank.iloc[0]["Ticker"]
    _plot_actual_vs_pred_lines("Ensemble", y_true_ens, y_pred_ens, best_ens_ticker)

# Call this wherever appropriate in your script:
make_spy_and_best_r2_plots()
