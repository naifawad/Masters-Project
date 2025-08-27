import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import randint, uniform
from functools import lru_cache
from sklearn.pipeline import Pipeline
from feature_engine.outliers import Winsorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


@lru_cache(maxsize=512)
def _dl_cached(tickers_key, start, end):
    # Keep Yahoo's MultiIndex columns (group_by='column' => level 0 = field, level 1 = ticker)
    return yf.download(
        tickers_key,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        group_by='column'
    )

def download_stock_data(tickers, start_date, end_date):
    """
    Returns the FULL Yahoo-style DataFrame (MultiIndex columns when tickers is a list),
    so callers can safely do df['Adj Close'] to get the wide matrix of adj closes.
    """
    key = tuple(tickers) if isinstance(tickers, (list, tuple)) else tickers
    data = _dl_cached(key, start_date, end_date)

    if data is None or data.empty:
        raise ValueError(f"No data returned for {tickers}.")

    if isinstance(data.columns, pd.MultiIndex):
        # Expect level 0 to contain fields like 'Adj Close'
        if 'Adj Close' not in data.columns.get_level_values(0):
            raise ValueError("Downloaded data has no 'Adj Close' field.")
        adj = data['Adj Close']
        # Ensure not all NaN
        if adj.isna().all().all():
            raise ValueError(f"'Adj Close' is entirely NaN for {tickers}.")
    else:
        # Single-ticker DataFrame
        if 'Adj Close' not in data.columns:
            raise ValueError("Downloaded data has no 'Adj Close' column.")
        if data['Adj Close'].isna().all():
            raise ValueError(f"'Adj Close' is entirely NaN for {tickers}.")

    return data  # callers will do ['Adj Close'] themselves




def generate_walkforward_ml_views(
    tickers, start_date, end_date, target_horizon=20,
    n_splits=5, n_iter_rs=20, random_state=42, price_data: dict = None, 
    macro_data: pd.DataFrame = None  , refit_on_full_data=True, target_mode='excess_spy'
):
    """
    Pooled cross-sectional learner with:
      - monthly technicals + 1M-lagged macro features
      - target: next-1M excess log return vs SPY
      - inner RandomizedSearchCV per fold (time-aware)
      - outer walk-forward (TimeSeriesSplit) for OOS R²
    Returns:
      investor_views: dict[ticker] -> predicted next 1M excess (log) return
      view_confidences: dict[ticker] -> scalar confidence (avg OOS R² floored at 0.1)
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import r2_score
    from scipy.stats import randint, uniform

    # --- small helpers (self-contained) ---
    def _to_monthly_close(df):
        m = pd.DataFrame()
        m['Adj Close'] = df['Adj Close'].resample('M').last()
        return m.dropna()

    def fetch_macro_monthly(start, end):
        macro_tickers = {
            'VIX': '^VIX', 'TNX': '^TNX', 'IRX': '^IRX',
            'WTI': 'CL=F', 'GOLD': 'GC=F', 'USD': 'UUP',
            'ACWX': 'ACWX', 'HYG': 'HYG', 'LQD': 'LQD'
        }
        raw = {}
        for k, t in macro_tickers.items():
            d = download_stock_data(t, start, end)  # daily
            d_m = _to_monthly_close(d).rename(columns={'Adj Close': k})
            raw[k] = d_m
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
        # strict: lag macros by 1 month
        feats = feats.shift(1).dropna()
        return feats

    def build_panel_with_macros(tickers, start, end, use_smooth_3m=False):
        if macro_data is not None:
            macro = macro_data.loc[start:end].shift(1).dropna()
        else:
            macro = fetch_macro_monthly(start, end)
        rows = []
        for t in tickers:
            if price_data and t in price_data:
                stock_d = price_data[t].loc[start:end]
            else:
                stock_d = download_stock_data(t, start, end)
            stock_m = _to_monthly_close(stock_d).rename(columns={'Adj Close':'PX'})
            df = stock_m.join(macro, how='outer')

            px  = df['PX']
            logp = np.log(px)
            r1  = logp.diff()

            mom12 = r1.rolling(12).sum() - r1
            mom6  = r1.rolling(6).sum() - r1
            rev1  = -r1
            vol3  = r1.rolling(3).std()
            vol6  = r1.rolling(6).std()
            vol12 = r1.rolling(12).std()
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

            # targets: next 1M log return, excess vs SPY
            if price_data and 'SPY' in price_data:
                spy_d = price_data['SPY'].loc[start:end]
            else:
                spy_d = download_stock_data('SPY', start, end)

            spy_m = _to_monthly_close(spy_d)
            spy_r1 = np.log(spy_m['Adj Close']).diff()

            if use_smooth_3m:
                tgt = r1.shift(-1).rolling(3).mean()
                spy_fwd = spy_r1.shift(-1).rolling(3).mean()
            else:
                tgt = r1.shift(-1)  # next month
                spy_fwd = spy_r1.shift(-1)

            if target_mode == 'raw':
                feat['target'] = tgt                  # raw forward return
            else:
                feat['target'] = tgt.sub(spy_fwd.reindex(feat.index), fill_value=0.0)  # excess vs SPY


            feat['Ticker'] = t
            rows.append(feat)

        panel = pd.concat(rows).sort_index()
        panel = panel.loc[(panel.index >= pd.to_datetime(start)) & (panel.index <= pd.to_datetime(end))]
        return panel.dropna()

    # --- build pooled panel ---
    panel = build_panel_with_macros(tickers, start_date, end_date, use_smooth_3m=False)
    y = panel['target'].copy()

 
    # features (+ one-hot ticker)
    X = panel.drop(columns=['target'])  # keep 'Ticker' column raw


    last_date = X.index.max()

   
    # =========================
    # CV + Walk-Forward
    # =========================
    outer = TimeSeriesSplit(n_splits=n_splits)
    inner = TimeSeriesSplit(n_splits=3)

    rf_param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(2, 12),
        'min_samples_split': randint(2, 12),
        'min_samples_leaf': randint(1, 8)
    }
    gb_param_dist = {
        'n_estimators': randint(50, 500),
        'learning_rate': uniform(0.01, 0.2),
        'max_depth': randint(2, 6),
        'subsample': uniform(0.6, 0.4),
        'min_samples_leaf': randint(1, 8),
        'max_features': ['sqrt', 'log2', None]
}

    
 
    def _fit_with_rs(base_model, param_dist, X_tr, y_tr):
        # figure out columns from the *training fold only*
        cat_cols = ['Ticker'] if 'Ticker' in X_tr.columns else []
        num_cols = [c for c in X_tr.columns if c not in cat_cols]

        # numeric branch = your existing pipeline steps
        num_pipe = Pipeline([
            ("var", VarianceThreshold(threshold=1e-12)),
            ("winsor", Winsorizer(
                capping_method="gaussian",
                tail="both",
                fold=3.0,
                missing_values="ignore"
            )),
            ("imputer", SimpleImputer(strategy="median")),
        ])

        transformers = [("num", num_pipe, num_cols)]
        if cat_cols:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat_cols)
            )

        pre = ColumnTransformer(transformers=transformers, remainder="drop")

        pipeline = Pipeline([
            ("pre", pre),
            ("model", base_model),
        ])

        # safe param grid
        param_dist = param_dist or {}
        param_dist_prefixed = {f"model__{k}": v for k, v in param_dist.items()}

        rs = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist_prefixed,
            n_iter=n_iter_rs,
            cv=inner,                  # your TimeSeriesSplit(n_splits=3)
            scoring="r2",
            n_jobs=-1,
            random_state=random_state,
            verbose=0,
            error_score="raise"        # helpful while debugging
        )
        rs.fit(X_tr, y_tr)
        return rs.best_estimator_


    def _ens(models, X_):
        return np.mean([m.predict(X_) for m in models], axis=0)

    r2_folds = []
    y_true_all, y_pred_all = [], []

    mdl_rf_last = None

    # Outer walk-forward
    for tr_idx, te_idx in outer.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

          # --- Purge one period to avoid fold-edge leakage (monthly horizon) ---
        if len(X_te) > 1:
            X_te = X_te.iloc[1:]
            y_te = y_te.iloc[1:]
            if len(X_te) == 0:
                continue
        
        # 1) RF with RS
        rf_base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        mdl_rf = _fit_with_rs(rf_base, rf_param_dist, X_tr, y_tr)
        # 2) GB with RS
        gb_base = GradientBoostingRegressor(random_state=random_state)
        mdl_gb  = _fit_with_rs(gb_base, gb_param_dist, X_tr, y_tr)

        # 3)Ensemble
        y_hat = _ens([mdl_rf, mdl_gb], X_te)
        r2_folds.append(r2_score(y_te, y_hat))
        y_true_all.append(y_te.values)
        y_pred_all.append(y_hat)

        # Save last fold models in case we skip final refit
        mdl_rf_last = mdl_rf

    # pooled OOS R² (not returned, but used for confidence)
    if r2_folds:
        avg_r2 = float(np.nanmean(r2_folds))
    else:
        avg_r2 = 0.1

    # FINAL: refit on full in-window data with RS to produce views
    if refit_on_full_data:
        mdl_rf_f = _fit_with_rs(
            RandomForestRegressor(random_state=random_state, n_jobs=-1),
            rf_param_dist, X, y
        )
        mdl_gb_f = _fit_with_rs(
            GradientBoostingRegressor(random_state=random_state),
            gb_param_dist, X, y
        )
    else:
        # Reuse last fold's best models
        mdl_rf_f = mdl_rf_last
        mdl_gb_f = mdl_gb

    latest_mask = (X.index == last_date)
    X_latest = X.loc[latest_mask]
    tickers_latest = panel.loc[latest_mask, 'Ticker'].values

    y_pred_latest = _ens([mdl_rf_f, mdl_gb_f], X_latest)

    investor_views = {tkr: float(pred) for tkr, pred in zip(tickers_latest, y_pred_latest)}

    conf_val = float(round(max(avg_r2, 0.1), 6))
    view_confidences = {t: conf_val for t in investor_views.keys()}

    # Fill any missing tickers with neutral view/conf
    for t in tickers:
        if t not in investor_views:
            investor_views[t] = 0.0
            view_confidences[t] = 0.55

    return investor_views

