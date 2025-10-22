#!/usr/bin/env python3
"""
ndvi_analysis.py

Updated: Streamlit visualization + fixes + enhanced statistics.

Features:
- Read CSV or Excel (local path or HTTP(S) link) including OneDrive direct-download handling.
- Descriptive stats
- Mann-Kendall (with tie correction) + Sen's slope (with CIs)
- Moving average, rolling regression slope, autocorrelation
- Change-point detection (ruptures: PELT and Binseg)
- Weighted least squares (cloud masking)
- Multivariate regression (statsmodels OLS) + diagnostics (VIF, Durbin-Watson, Cook's D, CI, p-values)
- PCA (sklearn) for multiband/multifeature inputs
- Export results to an Excel file and plots to a folder
- CLI with arguments and Streamlit interactive UI

Usage:
    CLI:
      python thesis/ndvi_analysis.py --input sample.csv --datecol "C0/date" --ndvicol "C0/mean" --outdir out

    Streamlit:
      streamlit run thesis/ndvi_analysis.py

Author: Generated for user (updated)
"""
from __future__ import annotations

import os
import sys
import argparse
import math
import io
from datetime import datetime
from typing import Optional, Sequence, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional libs
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATS_MODELS = True
except Exception:
    STATS_MODELS = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except Exception:
    RUPTURES_AVAILABLE = False

try:
    from scipy import stats
    SCIPY = True
except Exception:
    SCIPY = False

# Streamlit and plotly for interactive UI
try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

# requests for fetching remote files (OneDrive)
try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

# -------------------- Utilities --------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _download_to_bytes(url: str) -> bytes:
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("requests is required to download remote files. pip install requests")
    # Attempt to follow redirects and stream content
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    return resp.content

def read_table(path_or_url: str,
               date_col: Optional[str] = None,
               ndvi_col: Optional[str] = None,
               date_col_hints: Tuple[str, ...] = ('date', 'C0/date', 'Date'),
               ndvi_hints: Tuple[str, ...] = ('ndvi_mean', 'C0/mean', 'ndvi', 'NDVI', 'C0_mean')) -> pd.DataFrame:
    """Read CSV or Excel from local path or URL and normalize column names.

    Supports direct OneDrive shared links by downloading content to memory first.
    """
    # decide whether it's a URL or local path
    is_url = str(path_or_url).lower().startswith(('http://', 'https://'))
    # convenience: convert some OneDrive short links to downloadable content via requests
    df: pd.DataFrame
    try:
        if is_url:
            # If it's an Excel file by extension or hint
            lower = path_or_url.lower()
            if lower.endswith(('.xls', '.xlsx')) or 'onedrive' in lower or '1drv.ms' in lower:
                # download bytes then feed to pandas
                content = _download_to_bytes(path_or_url)
                bio = io.BytesIO(content)
                df = pd.read_excel(bio)
            else:
                # try CSV first; if fails, try excel
                try:
                    content = _download_to_bytes(path_or_url)
                    # Try to decode as text for read_csv
                    txt = content.decode('utf-8', errors='replace')
                    df = pd.read_csv(io.StringIO(txt))
                except Exception:
                    # fallback to excel read
                    bio = io.BytesIO(content)
                    df = pd.read_excel(bio)
        else:
            # local file
            if str(path_or_url).lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(path_or_url)
            else:
                df = pd.read_csv(path_or_url)
    except Exception as exc:
        raise RuntimeError(f"Failed to read input {path_or_url}: {exc}")

    df = _normalize_columns(df)

    # find date column
    if date_col:
        if date_col not in df.columns:
            raise ValueError(f"Provided date column '{date_col}' not found in file.")
        date_col_name = date_col
    else:
        date_col_name = None
        for hint in date_col_hints:
            if hint in df.columns:
                date_col_name = hint; break
        if date_col_name is None:
            # fuzzy find
            for c in df.columns:
                if 'date' in str(c).lower():
                    date_col_name = c; break
    if date_col_name is None:
        raise ValueError("No date column found. Provide --datecol.")
    df['date'] = pd.to_datetime(df[date_col_name], errors='coerce')
    if df['date'].isna().any():
        # try infer
        df['date'] = pd.to_datetime(df[date_col_name], infer_datetime_format=True, errors='coerce')

    # find ndvi column
    if ndvi_col:
        if ndvi_col not in df.columns:
            raise ValueError(f"Provided ndvi column '{ndvi_col}' not found in file.")
        ndvi_col_name = ndvi_col
    else:
        ndvi_col_name = None
        for hint in ndvi_hints:
            if hint in df.columns:
                ndvi_col_name = hint; break
        if ndvi_col_name is None:
            for c in df.columns:
                cn = str(c).lower()
                if ('ndvi' in cn) or ('mean' in cn and 'c0' in cn):
                    ndvi_col_name = c; break
    if ndvi_col_name is None:
        raise ValueError("No NDVI column found. Provide --ndvicol.")
    df['ndvi_mean'] = pd.to_numeric(df[ndvi_col_name], errors='coerce')

    df = df.sort_values('date').reset_index(drop=True)
    return df

def days_since_first(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'date' not in df.columns:
        raise ValueError("DataFrame missing 'date' column")
    df['t_days'] = (df['date'] - df['date'].iat[0]).dt.total_seconds() / 86400.0
    return df

# -------------------- Descriptive stats --------------------
def descriptive_stats(series: pd.Series) -> Dict[str, Any]:
    s = series.dropna()
    n = int(s.count())
    mean = float(s.mean()) if n else np.nan
    median = float(s.median()) if n else np.nan
    std = float(s.std(ddof=1)) if n > 1 else np.nan
    var = float(s.var(ddof=1)) if n > 1 else np.nan
    cv = (std / mean) * 100 if mean not in (0, None, np.nan) and not math.isnan(std) else np.nan
    skew = float(s.skew()) if n > 0 else np.nan
    kurt = float(s.kurtosis()) if n > 0 else np.nan
    p10 = float(s.quantile(0.1)) if n > 0 else np.nan
    p90 = float(s.quantile(0.9)) if n > 0 else np.nan
    iqr = float(s.quantile(0.75) - s.quantile(0.25)) if n > 0 else np.nan
    return {'n': n, 'mean': mean, 'median': median, 'std': std, 'var': var,
            'cv_percent': cv, 'skew': skew, 'kurtosis': kurt, 'p10': p10, 'p90': p90, 'iqr': iqr}

# -------------------- Mann-Kendall with tie correction --------------------
def mann_kendall_test(x: Sequence[float], times: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """
    Mann-Kendall S, Var(S) with tie correction, Z, p-value (normal approx), and Kendall's tau.
    x: 1D array-like of values
    times: optional times (e.g., days) corresponding to x to compute tau wrt time
    """
    arr = np.array([v for v in x if not pd.isna(v)])
    n = arr.size
    if n < 3:
        return {'S': np.nan, 'VarS': np.nan, 'Z': np.nan, 'p': np.nan, 'tau': np.nan}
    S = 0
    # compute S
    for i in range(n-1):
        diffs = np.sign(arr[i+1:] - arr[i])
        S += int(np.sum(diffs))
    # tie counts
    unique, counts = np.unique(arr, return_counts=True)
    tie_counts = counts[counts > 1]
    # Var(S) with tie correction
    var_s = (n*(n-1)*(2*n+5) - np.sum(tie_counts*(tie_counts-1)*(2*tie_counts+5))) / 18.0
    # Z value
    if var_s == 0:
        Z = 0.0
    else:
        if S > 0:
            Z = (S - 1) / math.sqrt(var_s)
        elif S < 0:
            Z = (S + 1) / math.sqrt(var_s)
        else:
            Z = 0.0
    # p-value
    if SCIPY:
        p = 2 * (1 - stats.norm.cdf(abs(Z)))
    else:
        p = 2 * (1 - 0.5*(1 + math.erf(abs(Z)/math.sqrt(2))))
    # Kendall's tau with respect to time/index
    try:
        if times is not None:
            tarr = np.array([v for v in times if not pd.isna(v)])
            if tarr.size == arr.size and SCIPY:
                kt = stats.kendalltau(tarr, arr)
                tau = float(kt.correlation)
            else:
                # fallback compute tau by index ordering
                if SCIPY:
                    kt = stats.kendalltau(np.arange(n), arr)
                    tau = float(kt.correlation)
                else:
                    tau = None
        else:
            if SCIPY:
                kt = stats.kendalltau(np.arange(n), arr)
                tau = float(kt.correlation)
            else:
                tau = None
    except Exception:
        tau = None
    return {'S': int(S), 'VarS': float(var_s), 'Z': float(Z), 'p': float(p), 'tau': tau}

# -------------------- Sen's slope --------------------
def sens_slope_with_ci(x: Sequence[float], t: Optional[Sequence[float]] = None, alpha: float = 0.05) -> Dict[str, Any]:
    arr = np.array([v for v in x if not pd.isna(v)])
    if t is None:
        tvals = np.arange(len(arr), dtype=float)
    else:
        tvals = np.array([v for v in t if not pd.isna(v)], dtype=float)
    n = len(arr)
    slopes = []
    for i in range(n-1):
        for j in range(i+1, n):
            dt = tvals[j] - tvals[i]
            if dt != 0:
                slopes.append((arr[j] - arr[i]) / dt)
    if len(slopes) == 0:
        return {'slope': np.nan, 'ci_low': np.nan, 'ci_high': np.nan, 'n_pairs': 0}
    slope_med = float(np.median(slopes))
    lo = float(np.percentile(slopes, 100*alpha/2))
    hi = float(np.percentile(slopes, 100*(1-alpha/2)))
    return {'slope': slope_med, 'ci_low': lo, 'ci_high': hi, 'n_pairs': len(slopes)}

# -------------------- Time series helpers --------------------
def moving_average(series: pd.Series, k: int = 3) -> pd.Series:
    return series.rolling(window=k, center=True, min_periods=1).mean()

def rolling_regression_slope(times: Sequence[float], values: Sequence[float], window: int = 5) -> np.ndarray:
    times = np.array(times, dtype=float)
    values = np.array(values, dtype=float)
    n = len(values)
    slopes = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        start = max(0, i-half)
        end = min(n, start + window)
        seg_t = times[start:end]
        seg_y = values[start:end]
        if len(seg_y) >= 2 and not np.all(np.isnan(seg_y)):
            # linear fit
            mask = ~np.isnan(seg_y)
            if mask.sum() >= 2:
                slope = np.polyfit(seg_t[mask], seg_y[mask], 1)[0]
                slopes[i] = slope
    return slopes

def autocorrelation(series: pd.Series, lag: int = 1) -> float:
    s = series.dropna()
    if len(s) <= lag:
        return float('nan')
    return float(s.autocorr(lag=lag))

# -------------------- Change point detection (ruptures) --------------------
def change_points_ruptures(values: Sequence[float], model: str = "l2", pen: Optional[float] = None, n_bkps: Optional[int] = None, method: str = 'pelt') -> list:
    """
    Use ruptures to detect change points. Returns breakpoints (ruptures format).
    """
    if not RUPTURES_AVAILABLE:
        raise RuntimeError("ruptures not available; pip install ruptures")
    arr = np.array(values).reshape(-1, 1)
    if method == 'pelt':
        algo = rpt.Pelt(model=model).fit(arr)
        if pen is None:
            pen = 3 * np.log(max(2, len(arr))) * np.var(arr)
        bkps = algo.predict(pen=pen)
    else:
        algo = rpt.Binseg(model=model).fit(arr)
        if n_bkps is None:
            n_bkps = 3
        bkps = algo.predict(n_bkps)
    return bkps

# -------------------- Weighted regression --------------------
def weighted_regression(times: Sequence[float], values: Sequence[float], weights: Sequence[float]) -> Dict[str, Any]:
    t = np.array(times, dtype=float)
    y = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    mask = (~np.isnan(y)) & (~np.isnan(t)) & (~np.isnan(w))
    if mask.sum() < 2:
        return {'slope': np.nan, 'intercept': np.nan, 'n': int(mask.sum())}
    t = t[mask]; y = y[mask]; w = w[mask]
    W = np.sum(w)
    tbar = np.sum(w * t) / W
    ybar = np.sum(w * y) / W
    num = np.sum(w * (t - tbar) * (y - ybar))
    den = np.sum(w * (t - tbar)**2)
    slope = num / den if den != 0 else np.nan
    intercept = ybar - slope * tbar
    return {'slope': float(slope), 'intercept': float(intercept), 'n': int(mask.sum())}

# -------------------- Multivariate regression + diagnostics --------------------
def multivariate_regression(df: pd.DataFrame, y_col: str, x_cols: Sequence[str], add_constant: bool = True) -> Dict[str, Any]:
    if not STATS_MODELS:
        raise RuntimeError("statsmodels required for multivariate regression. pip install statsmodels")
    X = df[list(x_cols)].astype(float)
    if add_constant:
        X = sm.add_constant(X)
    y = df[y_col].astype(float)
    model = sm.OLS(y, X, missing='drop')
    res = model.fit()
    # Prepare VIF (exclude constant)
    vif = {}
    try:
        X_noconst = X.loc[:, [c for c in X.columns if c.lower() not in ('const', 'constant', 'intercept')]]
        for i, col in enumerate(X_noconst.columns):
            try:
                vif[col] = float(variance_inflation_factor(X_noconst.values, i))
            except Exception:
                vif[col] = np.nan
    except Exception:
        vif = {}
    # Cook's distance
    try:
        infl = res.get_influence()
        cooks = np.asarray(infl.cooks_distance[0])
    except Exception:
        cooks = np.array([])
    # Durbin-Watson
    try:
        dw = float(sm.stats.stattools.durbin_watson(res.resid))
    except Exception:
        dw = np.nan
    # Extract coefficients, p-values, conf_int
    params = res.params.to_dict()
    pvalues = res.pvalues.to_dict()
    conf_int_df = res.conf_int()
    conf_int = {k: (float(conf_int_df.loc[k, 0]), float(conf_int_df.loc[k, 1])) for k in conf_int_df.index}
    return {
        'model': res,
        'params': params,
        'pvalues': pvalues,
        'conf_int': conf_int,
        'r_squared': float(res.rsquared),
        'adj_r_squared': float(res.rsquared_adj),
        'vif': vif,
        'cooks_d': cooks,
        'durbin_watson': dw,
    }

# -------------------- PCA --------------------
def run_pca(df: pd.DataFrame, feature_cols: Sequence[str], n_components: int = 3, scale: bool = True) -> Dict[str, Any]:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required for PCA. pip install scikit-learn")
    X = df[list(feature_cols)].astype(float).dropna()
    if X.shape[0] == 0:
        raise ValueError("No rows without NaNs in the selected PCA features.")
    scaler = StandardScaler() if scale else None
    if scale:
        Xs = scaler.fit_transform(X)
    else:
        Xs = X.values
    pca = PCA(n_components=min(n_components, Xs.shape[1]))
    pcs = pca.fit_transform(Xs)
    loadings = pca.components_.T
    explained = pca.explained_variance_ratio_
    return {'pca': pca, 'pcs': pcs, 'loadings': loadings, 'explained_variance_ratio': explained, 'index': X.index, 'columns': X.columns}

# -------------------- Derivative & integral --------------------
def derivative(times: Sequence[float], values: Sequence[float]) -> np.ndarray:
    t = np.array(times, dtype=float)
    y = np.array(values, dtype=float)
    dt = np.diff(t)
    dy = np.diff(y)
    with np.errstate(divide='ignore', invalid='ignore'):
        dydt = dy / dt
    return dydt

def integral_trapezoid(times: Sequence[float], values: Sequence[float]) -> float:
    t = np.array(times, dtype=float)
    y = np.array(values, dtype=float)
    return float(np.trapz(y, t))

# -------------------- Reporting & plotting --------------------
def save_summary_to_excel(outpath: str, summary_dict: Dict[str, Any], dataframe: Optional[pd.DataFrame] = None):
    with pd.ExcelWriter(outpath) as writer:
        # summary dict as flattened DataFrame
        sflat = {}
        for k, v in summary_dict.items():
            # simple types
            try:
                sflat[k] = str(v) if not isinstance(v, (int, float, dict, list)) else v
            except Exception:
                sflat[k] = str(v)
        pd.DataFrame([sflat]).T.to_excel(writer, sheet_name='summary')
        if dataframe is not None:
            dataframe.to_excel(writer, sheet_name='data', index=False)

def plot_time_series_plotly(df: pd.DataFrame, show_ma: bool = True, trend: Optional[pd.Series] = None, bkps: Optional[Sequence[int]] = None) -> go.Figure:
    fig = px.line(df, x='date', y='ndvi_mean', markers=True, title='NDVI time series')
    if show_ma and 'ma3' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['ma3'], mode='lines', name='MA(3)', line=dict(dash='dash')))
    if trend is not None:
        fig.add_trace(go.Scatter(x=df['date'], y=trend, mode='lines', name='Trend (OLS)', line=dict(color='red')))
    # add Sen's slope annotation if present
    if 'sens_slope' in df.attrs:
        ss = df.attrs.get('sens_slope')
        if ss and 'slope' in ss:
            fig.add_annotation(text=f"Sen's slope: {ss['slope']:.6f}/day", xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False)
    # change-points
    if bkps:
        for b in bkps:
            if b <= 0 or b > len(df):
                continue
            dt = df['date'].iat[b-1]
            fig.add_vline(x=dt, line=dict(color='orange', width=2, dash='dot'))
    fig.update_layout(xaxis_title='Date', yaxis_title='NDVI', hovermode='x unified')
    return fig

# -------------------- Core analysis pipeline --------------------
def analyze(df: pd.DataFrame, cloud_col: Optional[str] = None, features: Optional[Sequence[str]] = None, pca_features: Optional[Sequence[str]] = None, pca_n: int = 3, run_ruptures: bool = True) -> Dict[str, Any]:
    df = df.copy()
    df = days_since_first(df)

    # cloud coverage
    if cloud_col and cloud_col in df.columns:
        df['cloudpct'] = pd.to_numeric(df[cloud_col], errors='coerce')
    elif 'cloudpct' not in df.columns:
        df['cloudpct'] = np.nan

    # Derived columns
    desc = descriptive_stats(df['ndvi_mean'])
    df['zscore'] = (df['ndvi_mean'] - desc['mean']) / desc['std'] if desc['std'] not in (None, np.nan, 0) else np.nan
    df['ma3'] = moving_average(df['ndvi_mean'], k=3)
    df['roll_slope_5'] = rolling_regression_slope(df['t_days'].values, df['ndvi_mean'].values, window=5)
    acf1 = autocorrelation(df['ndvi_mean'], lag=1)
    mk = mann_kendall_test(df['ndvi_mean'].values, times=df['t_days'].values)
    ss = sens_slope_with_ci(df['ndvi_mean'].values, t=df['t_days'].values)
    df.attrs['sens_slope'] = ss

    # Weighted regression (cloud weights)
    df['weight'] = 1.0
    if 'cloudpct' in df.columns:
        df['weight'] = 100.0 - df['cloudpct'].fillna(0.0)
        df['weight'] = df['weight'].clip(lower=0.1)  # avoid zeros

    wreg = weighted_regression(df['t_days'].values, df['ndvi_mean'].values, df['weight'].values)

    # OLS over time (simple)
    ols_res = None
    try:
        if STATS_MODELS:
            # simple model: ndvi_mean ~ t_days
            ols_res = multivariate_regression(df, 'ndvi_mean', ['t_days'])
    except Exception:
        ols_res = None

    # change points via ruptures
    bkps = None
    if run_ruptures and RUPTURES_AVAILABLE:
        try:
            bkps = change_points_ruptures(df['ndvi_mean'].values, method='pelt', pen=None)
        except Exception:
            bkps = None

    # Multivariate regression if features provided
    mv_res = None
    if features:
        feat_cols = list(features)
        try:
            mv_res = multivariate_regression(df, 'ndvi_mean', feat_cols)
        except Exception as e:
            mv_res = {'error': str(e)}

    # PCA
    pca_res = None
    if pca_features:
        try:
            pca_res = run_pca(df, pca_features, n_components=pca_n)
        except Exception as e:
            pca_res = {'error': str(e)}

    # Derivative & integral
    df['dydt'] = np.nan
    if len(df) >= 2:
        d1 = derivative(df['t_days'].values, df['ndvi_mean'].values)
        df.loc[df.index[1:], 'dydt'] = d1
    total_greenness = integral_trapezoid(df['t_days'].values, df['ndvi_mean'].values)

    result = {
        'df': df,
        'desc': desc,
        'acf1': acf1,
        'mann_kendall': mk,
        'sens_slope': ss,
        'weighted_regression': wreg,
        'ols': ols_res,
        'multivariate': mv_res,
        'pca': pca_res,
        'ruptures_bkps': bkps,
        'total_greenness': total_greenness,
    }
    return result

# -------------------- Streamlit app --------------------
def streamlit_app(default_input: Optional[str] = None):
    st.set_page_config(layout="wide", page_title="NDVI Analysis")
    st.title("NDVI Analysis Toolkit (Interactive)")

    with st.sidebar:
        st.header("Data input")
        input_type = st.radio("Input type", ("URL / path", "Upload file"))
        uploaded_df = None
        path_or_url = default_input or ""
        if input_type == "URL / path":
            path_or_url = st.text_input("Path or URL to CSV/XLSX (OneDrive links ok)", value=path_or_url)
        else:
            upl = st.file_uploader("Upload CSV or XLSX", type=['csv', 'xls', 'xlsx'])
            if upl is not None:
                try:
                    if upl.name.lower().endswith(('.xls', '.xlsx')):
                        uploaded_df = pd.read_excel(upl)
                    else:
                        uploaded_df = pd.read_csv(upl)
                except Exception as e:
                    st.error(f"Failed to read uploaded file: {e}")

        datecol = st.text_input("Date column (optional)", value="")
        ndvicol = st.text_input("NDVI column (optional)", value="")
        cloudcol = st.text_input("Cloud coverage column (optional)", value="C0/cloudCoveragePercent")
        features_text = st.text_input("Features for regression (comma-separated)", value="")
        pca_text = st.text_input("PCA features (comma-separated)", value="")
        pca_n = st.number_input("PCA components", min_value=1, max_value=10, value=3)
        run_ruptures = st.checkbox("Run change-point detection (ruptures)", value=RUPTURES_AVAILABLE)
        outdir = st.text_input("Output folder (server)", value="output")
        st.markdown("---")
        st.write("Libraries available:")
        st.write({
            'statsmodels': STATS_MODELS,
            'scikit-learn': SKLEARN_AVAILABLE,
            'ruptures': RUPTURES_AVAILABLE,
            'scipy': SCIPY,
            'streamlit': STREAMLIT_AVAILABLE,
            'requests': REQUESTS_AVAILABLE
        })
        run_btn = st.button("Run analysis")

    # load data
    df = None
    if uploaded_df is not None:
        df = _normalize_columns(uploaded_df)
    elif path_or_url:
        try:
            df = read_table(path_or_url, date_col=datecol or None, ndvi_col=ndvicol or None)
        except Exception as e:
            st.error(f"Failed loading data: {e}")
            df = None

    if df is None:
        st.info("Please provide a path/URL or upload a file to start.")
        return

    # Show dataset
    st.subheader("Data preview")
    st.dataframe(df.head(200))

    # Run analysis when requested or automatically
    if run_btn or st.sidebar.button("Run now"):
        features = [f.strip() for f in features_text.split(',') if f.strip()] if features_text else None
        pca_feats = [f.strip() for f in pca_text.split(',') if f.strip()] if pca_text else None
        try:
            res = analyze(df, cloud_col=(cloudcol or None), features=features, pca_features=pca_feats, pca_n=pca_n, run_ruptures=run_ruptures)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return

        out_df = res['df']
        # add sens slope to attrs for plotting
        out_df.attrs['sens_slope'] = res.get('sens_slope')

        # summary and metrics
        st.subheader("Descriptive statistics")
        st.json(res['desc'])

        st.subheader("Key metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("N", res['desc']['n'])
        col1.metric("Mean NDVI", f"{res['desc']['mean']:.4f}" if not math.isnan(res['desc']['mean']) else "NA")
        col2.metric("Std dev", f"{res['desc']['std']:.4f}" if not math.isnan(res['desc']['std']) else "NA")
        col2.metric("ACF(1)", f"{res['acf1']:.4f}" if not math.isnan(res['acf1']) else "NA")
        col3.metric("Mann-Kendall p", f"{res['mann_kendall']['p']:.4f}" if not math.isnan(res['mann_kendall']['p']) else "NA")
        col3.metric("Sen's slope (/day)", f"{res['sens_slope']['slope']:.6f}" if res['sens_slope'] and not math.isnan(res['sens_slope']['slope']) else "NA")

        # Time series interactive plot
        st.subheader("Time series")
        trend = None
        if res.get('ols') and isinstance(res['ols'], dict):
            # multivariate_regression returns dict containing 'params' etc
            try:
                params = res['ols']['params']
                slope = params.get('t_days', None)
                intercept = params.get('const', 0.0)
                if slope is not None:
                    trend = intercept + slope * out_df['t_days']
            except Exception:
                trend = None
        fig_ts = plot_time_series_plotly(out_df, show_ma=True, trend=trend, bkps=res.get('ruptures_bkps'))
        st.plotly_chart(fig_ts, use_container_width=True)

        # distribution
        st.subheader("Distribution")
        fig_hist = px.histogram(out_df, x='ndvi_mean', nbins=40, marginal='box', title='NDVI Distribution')
        st.plotly_chart(fig_hist, use_container_width=True)

        # autocorrelation plot (lags)
        st.subheader("Autocorrelation")
        maxlag = min(30, max(3, len(out_df)//2))
        acfs = [autocorrelation(out_df['ndvi_mean'], lag=l) for l in range(1, maxlag+1)]
        fig_acf = px.bar(x=list(range(1, maxlag+1)), y=acfs, labels={'x':'Lag', 'y':'ACF'}, title='Autocorrelation function')
        st.plotly_chart(fig_acf, use_container_width=True)

        # PCA visualization
        if res.get('pca') and isinstance(res['pca'], dict) and 'pcs' in res['pca']:
            st.subheader("PCA")
            pca = res['pca']
            pcs = pca['pcs']
            if pcs.shape[1] >= 2:
                pca_df = pd.DataFrame(pcs[:, :2], columns=['PC1', 'PC2'])
                pca_df['index'] = pca['index'].astype(str)
                fig_pca = px.scatter(pca_df, x='PC1', y='PC2', hover_name='index', title='PCA PC1 vs PC2',
                                    labels={'PC1':'PC1', 'PC2':'PC2'})
                st.plotly_chart(fig_pca, use_container_width=True)
            st.write("Explained variance ratio:", pca['explained_variance_ratio'])

        # Regression diagnostics
        st.subheader("Regression (time ~ NDVI) diagnostics")
        if res.get('ols'):
            ols_info = res['ols']
            # if dict from multivariate_regression
            if isinstance(ols_info, dict):
                st.write("R-squared:", ols_info.get('r_squared'))
                st.write("Adjusted R-squared:", ols_info.get('adj_r_squared'))
                st.write("Coefficients:")
                st.json({'params': ols_info.get('params'), 'pvalues': ols_info.get('pvalues'), 'conf_int': ols_info.get('conf_int')})
                st.write("VIF:", ols_info.get('vif'))
                st.write("Durbin-Watson:", ols_info.get('durbin_watson'))
                if isinstance(ols_info.get('cooks_d'), (list, np.ndarray)) and len(ols_info.get('cooks_d')) > 0:
                    cooks = np.array(ols_info.get('cooks_d'))
                    fig_cook = px.scatter(x=list(range(len(cooks))), y=cooks, labels={'x':'observation', 'y':"Cook's D"}, title="Cook's D")
                    st.plotly_chart(fig_cook, use_container_width=True)
            else:
                # statsmodels result object
                st.text(ols_info.summary().as_text())

        # save outputs
        if st.button("Save outputs to disk"):
            os.makedirs(outdir, exist_ok=True)
            out_excel = os.path.join(outdir, 'ndvi_report.xlsx')
            summary = {
                'desc': res['desc'],
                'acf1': res['acf1'],
                'mann_kendall': res['mann_kendall'],
                'sens_slope': res['sens_slope'],
                'weighted_regression': res['weighted_regression'],
                'total_greenness': res['total_greenness'],
                'ruptures_bkps': res['ruptures_bkps'],
            }
            try:
                save_summary_to_excel(out_excel, summary, dataframe=out_df)
                # save plots as png
                fig_ts.write_image(os.path.join(outdir, 'ndvi_timeseries.png'))
                fig_hist.write_image(os.path.join(outdir, 'ndvi_hist.png'))
                st.success(f"Saved Excel and plots to {outdir}")
            except Exception as e:
                st.error(f"Failed to save outputs: {e}")

# -------------------- CLI main --------------------
def main_cli(args):
    df = read_table(args.input, date_col=args.datecol or None, ndvi_col=args.ndvicol or None)
    res = analyze(df, cloud_col=(args.cloudcol or None), features=(args.features.split(',') if args.features else None),
                  pca_features=(args.pca_features.split(',') if args.pca_features else None), pca_n=args.pca_n,
                  run_ruptures=True)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Save outputs
    summary = {
        'desc': res['desc'],
        'acf1': res['acf1'],
        'mann_kendall': res['mann_kendall'],
        'sens_slope': res['sens_slope'],
        'weighted_regression': res['weighted_regression'],
        'total_greenness': res['total_greenness'],
        'ruptures_bkps': res['ruptures_bkps'],
    }
    out_excel = os.path.join(outdir, args.outfile if args.outfile.endswith('.xlsx') else (args.outfile + '.xlsx'))
    save_summary_to_excel(out_excel, summary, dataframe=res['df'])
    # quick plot with OLS trend line using plotly for saving
    trend = None
    if res.get('ols') and isinstance(res['ols'], dict):
        try:
            params = res['ols']['params']
            slope = params.get('t_days', None)
            intercept = params.get('const', 0.0)
            if slope is not None:
                trend = intercept + slope * res['df']['t_days']
        except Exception:
            trend = None
    fig = plot_time_series_plotly(res['df'], show_ma=True, trend=trend, bkps=res.get('ruptures_bkps'))
    png_path = os.path.join(outdir, 'ndvi_timeseries.png')
    try:
        fig.write_image(png_path)
    except Exception:
        # fallback to matplotlib
        plot_path = os.path.join(outdir, 'ndvi_timeseries_mpl.png')
        plt.figure(figsize=(10, 4))
        plt.plot(res['df']['date'], res['df']['ndvi_mean'], marker='o')
        if 'ma3' in res['df'].columns:
            plt.plot(res['df']['date'], res['df']['ma3'], linestyle='--')
        plt.title("NDVI time series")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    res['df'].to_csv(os.path.join(outdir, 'ndvi_derived.csv'), index=False)

    # Print summary to console
    print("Summary saved to:", out_excel)
    print("Key metrics:")
    print("  mean:", res['desc']['mean'], "std:", res['desc']['std'], "n:", res['desc']['n'])
    mk = res['mann_kendall']
    print("  Mann-Kendall S:", mk['S'], "Z:", mk['Z'], "p:", mk['p'])
    ss = res['sens_slope']
    print("  Sen's slope (per day):", ss['slope'], "CI:", (ss['ci_low'], ss['ci_high']))
    print("  Weighted slope (cloud weights):", res['weighted_regression'].get('slope'))
    if res.get('ruptures_bkps'):
        print("  Change-points (ruptures):", res['ruptures_bkps'])
    if res.get('multivariate'):
        mv = res['multivariate']
        print("  Multivariate regression result:", "error" in mv and mv['error'] or "OK")
    if res.get('pca'):
        pca_res = res['pca']
        if isinstance(pca_res, dict) and 'explained_variance_ratio' in pca_res:
            print("  PCA explained variance:", pca_res['explained_variance_ratio'])
    print("  Total greenness (NDVI·days):", res['total_greenness'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NDVI analysis toolkit")
    parser.add_argument('--input', required=False, help='Path or URL to CSV/XLSX with NDVI table')
    parser.add_argument('--datecol', default=None, help='Name of date column if not auto-detected')
    parser.add_argument('--ndvicol', default=None, help='Name of NDVI column if not auto-detected')
    parser.add_argument('--cloudcol', default='C0/cloudCoveragePercent', help='Cloud coverage column name (optional)')
    parser.add_argument('--features', default=None, help='Comma-separated feature columns for multivariate regression')
    parser.add_argument('--pca_features', default=None, help='Comma-separated columns for PCA')
    parser.add_argument('--pca_n', type=int, default=3, help='Number of PCA components')
    parser.add_argument('--outdir', default='output', help='Folder to write results and plots')
    parser.add_argument('--outfile', default='ndvi_report.xlsx', help='Output Excel filename')
    parser.add_argument('--streamlit', action='store_true', help='Launch streamlit UI (if streamlit is available)')
    args = parser.parse_args()

    # If streamlit is available and requested (or running under streamlit), launch app
    if (args.streamlit or ('STREAMLIT_RUNNING' in os.environ) or STREAMLIT_AVAILABLE and any('streamlit' in arg for arg in sys.argv)):
        if not STREAMLIT_AVAILABLE:
            print("Streamlit is not installed. To use the interactive UI install streamlit and plotly.")
            sys.exit(1)
        # If an input is provided, pass as default to UI
        default_input = args.input if args.input else None
        streamlit_app(default_input=default_input)
    else:
        if not args.input:
            print("No input provided. Use --input PATH_OR_URL or run with --streamlit for interactive mode.")
            parser.print_help()
            sys.exit(1)
        main_cli(args)
