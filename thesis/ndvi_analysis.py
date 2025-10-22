#!/usr/bin/env python3
"""
ndvi_analysis.py

Sentinel-2 NDVI analysis toolkit ready for GitHub deployment.

Features:
- Read CSV or Excel (local path or HTTP(S) link) with expected columns:
  Date column (auto-detected: 'date' or 'C0/date') and NDVI column 'ndvi_mean' or 'C0/mean'
- Descriptive stats
- Mann-Kendall (with tie correction) + Sen's slope (with CIs)
- Moving average, rolling regression slope, autocorrelation
- Change-point detection (ruptures: PELT and Binseg)
- Weighted least squares (cloud masking)
- Multivariate regression (statsmodels OLS) + diagnostics (VIF, Durbin-Watson, Cook's D)
- PCA (sklearn) for multiband/multifeature inputs
- Export results to an Excel file and plots to a folder
- CLI with arguments

Usage:
  python ndvi_analysis.py --input sample.csv --datecol "C0/date" --ndvicol "C0/mean" --outdir out

Author: Generated for user
"""

import os
import sys
import argparse
import math
from datetime import datetime
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

# -------------------- Utilities --------------------
def read_table(path_or_url, date_col_hints=('date','C0/date','Date'), ndvi_hints=('ndvi_mean','C0/mean','ndvi','NDVI','C0_mean')):
    """Read CSV or Excel from local path or URL and normalize column names."""
    # decide read_csv or read_excel by extension or content-type
    if str(path_or_url).lower().endswith(('.xls','.xlsx')):
        df = pd.read_excel(path_or_url)
    else:
        df = pd.read_csv(path_or_url)
    # normalize columns
    lc = {c:c.strip() for c in df.columns}
    df.columns = list(lc.keys())
    # find date column
    date_col = None
    for hint in date_col_hints:
        if hint in df.columns:
            date_col = hint
            break
    if date_col is None:
        # try fuzzy find
        for c in df.columns:
            if 'date' in c.lower():
                date_col = c; break
    if date_col is None:
        raise ValueError("No date column found. Provide --datecol.")
    df['date'] = pd.to_datetime(df[date_col])
    # find ndvi column
    ndvi_col = None
    for hint in ndvi_hints:
        if hint in df.columns:
            ndvi_col = hint; break
    if ndvi_col is None:
        for c in df.columns:
            if 'ndvi' in c.lower() or 'mean' in c.lower() and 'c0' in c.lower():
                ndvi_col = c; break
    if ndvi_col is None:
        raise ValueError("No NDVI column found. Provide --ndvicol.")
    df['ndvi_mean'] = pd.to_numeric(df[ndvi_col], errors='coerce')
    return df.sort_values('date').reset_index(drop=True)

def days_since_first(df):
    df = df.copy()
    df['t_days'] = (df['date'] - df['date'].iat[0]).dt.total_seconds()/86400.0
    return df

# -------------------- Descriptive stats --------------------
def descriptive_stats(series):
    n = series.count()
    mean = series.mean()
    median = series.median()
    std = series.std(ddof=1)
    var = series.var(ddof=1)
    cv = (std / mean) * 100 if mean != 0 else np.nan
    skew = series.skew()
    kurt = series.kurtosis()
    p10 = series.quantile(0.1)
    p90 = series.quantile(0.9)
    return {'n':n, 'mean':mean, 'median':median, 'std':std, 'var':var, 'cv_percent':cv, 'skew':skew, 'kurtosis':kurt, 'p10':p10, 'p90':p90}

# -------------------- Mann-Kendall with tie correction --------------------
def mann_kendall_test(x):
    """
    Mann-Kendall S, Var(S) with tie correction, Z, p-value (normal approx), and Kendall's tau.
    x: 1D array-like
    """
    arr = np.array([v for v in x if not pd.isna(v)])
    n = arr.size
    if n < 3:
        return {'S':np.nan, 'VarS':np.nan, 'Z':np.nan, 'p':np.nan, 'tau':np.nan}
    S = 0
    # count ties for VarS correction
    ties = {}
    for i in range(n-1):
        for j in range(i+1, n):
            if arr[j] > arr[i]:
                S += 1
            elif arr[j] < arr[i]:
                S -= 1
    # tie counts
    unique, counts = np.unique(arr, return_counts=True)
    tie_counts = counts[counts > 1]
    # Var(S) with tie correction
    var_s = (n*(n-1)*(2*n+5) - np.sum(tie_counts*(tie_counts-1)*(2*tie_counts+5))) / 18.0
    # Z
    if S > 0:
        Z = (S - 1) / math.sqrt(var_s)
    elif S < 0:
        Z = (S + 1) / math.sqrt(var_s)
    else:
        Z = 0.0
    if SCIPY:
        p = 2 * (1 - stats.norm.cdf(abs(Z)))
    else:
        p = 2 * (1 - 0.5*(1 + math.erf(abs(Z)/math.sqrt(2))))
    # Kendall tau (scipy if available)
    if SCIPY:
        kt = stats.kendalltau(np.arange(n), arr)  # note: better to compute tau(arr, time) but here time is index
        tau = kt.correlation
    else:
        tau = None
    return {'S':S, 'VarS':var_s, 'Z':Z, 'p':p, 'tau':tau}

# -------------------- Sen's slope --------------------
def sens_slope_with_ci(x, t=None, alpha=0.05):
    arr = np.array([v for v in x if not pd.isna(v)])
    if t is None:
        tvals = np.arange(len(arr))
    else:
        tvals = np.array([v for v in t if not pd.isna(v)])
    slopes = []
    n = len(arr)
    for i in range(n-1):
        for j in range(i+1, n):
            dt = tvals[j] - tvals[i]
            if dt != 0:
                slopes.append((arr[j] - arr[i]) / dt)
    if len(slopes) == 0:
        return {'slope':np.nan, 'ci_low':np.nan, 'ci_high':np.nan}
    slope_med = np.median(slopes)
    lo = np.percentile(slopes, 100*alpha/2)
    hi = np.percentile(slopes, 100*(1-alpha/2))
    return {'slope':slope_med, 'ci_low':lo, 'ci_high':hi, 'n_pairs':len(slopes)}

# -------------------- Time series helpers --------------------
def moving_average(series, k=3):
    return series.rolling(window=k, center=True, min_periods=1).mean()

def rolling_regression_slope(times, values, window=5):
    n = len(values)
    slopes = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        start = max(0, i-half)
        end = min(n, start + window)
        if end - start >= 2:
            tseg = times[start:end]
            yseg = values[start:end]
            slope = np.polyfit(tseg, yseg, 1)[0]
            slopes[i] = slope
    return slopes

def autocorrelation(series, lag=1):
    s = series.dropna()
    if len(s) <= lag:
        return np.nan
    return s.autocorr(lag=lag)

# -------------------- Change point detection (ruptures) --------------------
def change_points_ruptures(values, model="l2", pen=None, n_bkps=None, method='pelt'):
    """
    Use ruptures to detect change points.
    method: 'pelt' or 'binseg'
    - pen: penalty value for pelt
    - n_bkps: number of breakpoints for binseg
    returns list of indices where segments end (ruptures format)
    """
    if not RUPTURES_AVAILABLE:
        raise RuntimeError("ruptures not available; pip install ruptures")
    arr = np.array(values).reshape(-1,1)
    if method == 'pelt':
        algo = rpt.Pelt(model=model).fit(arr)
        if pen is None:
            # default penalty heuristic
            pen = 3 * np.log(len(arr)) * np.var(arr)
        bkps = algo.predict(pen=pen)
    else:
        algo = rpt.Binseg(model=model).fit(arr)
        if n_bkps is None:
            n_bkps = 3
        bkps = algo.predict(n_bkps)
    return bkps

# -------------------- Weighted regression --------------------
def weighted_regression(times, values, weights):
    t = np.array(times)
    y = np.array(values)
    w = np.array(weights)
    mask = (~np.isnan(y)) & (~np.isnan(t)) & (~np.isnan(w))
    if mask.sum() < 2:
        return {'slope':np.nan, 'intercept':np.nan}
    t = t[mask]; y = y[mask]; w = w[mask]
    W = np.sum(w)
    tbar = np.sum(w * t) / W
    ybar = np.sum(w * y) / W
    num = np.sum(w * (t - tbar) * (y - ybar))
    den = np.sum(w * (t - tbar)**2)
    slope = num / den if den != 0 else np.nan
    intercept = ybar - slope * tbar
    return {'slope':slope, 'intercept':intercept}

# -------------------- Multivariate regression + diagnostics --------------------
def multivariate_regression(df, y_col, x_cols, add_constant=True):
    if not STATS_MODELS:
        raise RuntimeError("statsmodels required for multivariate regression. pip install statsmodels")
    X = df[x_cols].astype(float)
    if add_constant:
        X = sm.add_constant(X)
    y = df[y_col].astype(float)
    model = sm.OLS(y, X, missing='drop')
    res = model.fit()
    # VIF
    vif = {}
    if X.shape[1] > 1:
        X_for_vif = X.drop(columns=[c for c in X.columns if c=='const' or c=='constant' or c=='intercept' and c in X.columns], errors='ignore')
        # need design matrix without constant for VIF
        for i, col in enumerate(X_for_vif.columns):
            try:
                vif[col] = variance_inflation_factor(X_for_vif.values, i)
            except Exception:
                vif[col] = np.nan
    # Cook's distance
    infl = res.get_influence()
    cooks = infl.cooks_distance[0]
    # Durbin-Watson
    dw = sm.stats.stattools.durbin_watson(res.resid)
    return {'model':res, 'vif':vif, 'cooks_d':cooks, 'durbin_watson':dw}

# -------------------- PCA --------------------
def run_pca(df, feature_cols, n_components=3, scale=True):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required for PCA. pip install scikit-learn")
    X = df[feature_cols].astype(float).dropna()
    scaler = StandardScaler() if scale else None
    if scale:
        Xs = scaler.fit_transform(X)
    else:
        Xs = X.values
    pca = PCA(n_components=min(n_components, Xs.shape[1]))
    pcs = pca.fit_transform(Xs)
    loadings = pca.components_.T
    explained = pca.explained_variance_ratio_
    return {'pca':pca, 'pcs':pcs, 'loadings':loadings, 'explained_variance_ratio':explained, 'columns':X.index}

# -------------------- Derivative & integral --------------------
def derivative(times, values):
    t = np.array(times)
    y = np.array(values)
    dt = np.diff(t)
    dy = np.diff(y)
    dydt = dy / dt
    return dydt

def integral_trapezoid(times, values):
    t = np.array(times)
    y = np.array(values)
    return np.trapz(y, t)

# -------------------- Reporting & plotting --------------------
def save_summary_to_excel(outpath, summary_dict, dataframe=None):
    with pd.ExcelWriter(outpath) as writer:
        pd.DataFrame([summary_dict]).T.to_excel(writer, sheet_name='summary')
        if dataframe is not None:
            dataframe.to_excel(writer, sheet_name='data', index=False)

def plot_time_series(df, outdir, trend=None):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(10,5))
    plt.plot(df['date'], df['ndvi_mean'], marker='o', label='NDVI mean')
    if 'ma3' in df.columns:
        plt.plot(df['date'], df['ma3'], marker='s', linestyle='--', label='3-pt MA')
    if trend is not None:
        plt.plot(df['date'], trend, label='Trend line')
    plt.xlabel('Date'); plt.ylabel('NDVI'); plt.title('NDVI time series')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    fpath = os.path.join(outdir, 'ndvi_timeseries.png')
    plt.savefig(fpath)
    plt.close()
    return fpath

# -------------------- CLI main --------------------
def main(args):
    df = read_table(args.input, date_col_hints=(args.datecol,) if args.datecol else None)
    df = days_since_first(df)
    if args.cloudcol and args.cloudcol in df.columns:
        df['cloudpct'] = pd.to_numeric(df[args.cloudcol], errors='coerce')
    elif 'cloudpct' not in df.columns:
        df['cloudpct'] = np.nan

    # Derived columns
    desc = descriptive_stats(df['ndvi_mean'])
    df['zscore'] = (df['ndvi_mean'] - desc['mean']) / desc['std'] if desc['std'] and not math.isnan(desc['std']) else np.nan
    df['ma3'] = moving_average(df['ndvi_mean'], k=3)
    df['roll_slope_5'] = rolling_regression_slope(df['t_days'].values, df['ndvi_mean'].values, window=5)
    acf1 = autocorrelation(df['ndvi_mean'], lag=1)
    mk = mann_kendall_test(df['ndvi_mean'])
    ss = sens_slope_with_ci(df['ndvi_mean'], t=df['t_days'])
    # Weighted regression
    if 'cloudpct' in df.columns:
        df['weight'] = 100.0 - df['cloudpct'].fillna(0.0)
    else:
        df['weight'] = 1.0
    wreg = weighted_regression(df['t_days'], df['ndvi_mean'], df['weight'])
    # OLS
    ols = None
    if STATS_MODELS:
        try:
            ols_res = multivariate_regression(df, 'ndvi_mean', ['t_days'])
            ols = ols_res['model']
        except Exception:
            ols = None
    # change points
    bkps = None
    if RUPTURES_AVAILABLE:
        try:
            bkps = change_points_ruptures(df['ndvi_mean'].values, method='pelt', pen=None)
        except Exception:
            bkps = None

    # Multivariate regression if user requested features
    mv_res = None
    if args.features:
        feat_cols = args.features.split(',')
        mv_res = None
        try:
            mv_res = multivariate_regression(df, 'ndvi_mean', feat_cols)
        except Exception as e:
            mv_res = {'error':str(e)}

    # PCA if requested and sklearn available
    pca_res = None
    if args.pca_features:
        try:
            pca_res = run_pca(df, args.pca_features.split(','), n_components=args.pca_n)
        except Exception as e:
            pca_res = {'error':str(e)}

    # Derivative & integral
    df['dydt'] = np.nan
    d1 = derivative(df['t_days'].values, df['ndvi_mean'].values)
    df.loc[df.index[1:], 'dydt'] = d1
    total_greenness = integral_trapezoid(df['t_days'].values, df['ndvi_mean'].values)

    # Save outputs
    os.makedirs(args.outdir, exist_ok=True)
    summary = {
        'desc': desc,
        'acf1': acf1,
        'mann_kendall': mk,
        'sens_slope': ss,
        'weighted_regression': wreg,
        'total_greenness': total_greenness,
        'ruptures_bkps': bkps,
    }
    # Write Excel summary and csv
    out_excel = os.path.join(args.outdir, args.outfile if args.outfile.endswith('.xlsx') else (args.outfile + '.xlsx'))
    save_summary_to_excel(out_excel, summary, dataframe=df)
    # quick plot with OLS trend line
    trend = None
    if ols is not None:
        slope = ols.params.get('t_days', None)
        intercept = ols.params.get('const', 0.0)
        if slope is not None:
            trend = slope * df['t_days'] + intercept
    plot_time_series(df, args.outdir, trend=trend)

    # save df csv
    df.to_csv(os.path.join(args.outdir, 'ndvi_derived.csv'), index=False)

    # Print summary to console
    print("Summary saved to:", out_excel)
    print("Key metrics:")
    print("  mean:", desc['mean'], "std:", desc['std'], "n:", desc['n'])
    print("  Mann-Kendall S:", mk['S'], "Z:", mk['Z'], "p:", mk['p'])
    print("  Sen's slope (per day):", ss['slope'], "CI:", (ss['ci_low'], ss['ci_high']))
    print("  Weighted slope (cloud weights):", wreg['slope'])
    if bkps:
        print("  Change-points (ruptures):", bkps)
    if mv_res:
        print("  Multivariate regression result:", "error" in mv_res and mv_res['error'] or "OK")
    if pca_res:
        print("  PCA result:", "error" in pca_res and pca_res['error'] or f"explained {pca_res['explained_variance_ratio']}")
    print("  Total greenness (NDVI·days):", total_greenness)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NDVI analysis toolkit")
    parser.add_argument('--input', required=True, help='Path or URL to CSV/XLSX with NDVI table')
    parser.add_argument('--datecol', default=None, help='Name of date column if not auto-detected')
    parser.add_argument('--ndvicol', default=None, help='Name of NDVI column if not auto-detected')
    parser.add_argument('--cloudcol', default='C0/cloudCoveragePercent', help='Cloud coverage column name (optional)')
    parser.add_argument('--features', default=None, help='Comma-separated feature columns for multivariate regression')
    parser.add_argument('--pca_features', default=None, help='Comma-separated columns for PCA')
    parser.add_argument('--pca_n', type=int, default=3, help='Number of PCA components')
    parser.add_argument('--outdir', default='output', help='Folder to write results and plots')
    parser.add_argument('--outfile', default='ndvi_report.xlsx', help='Output Excel filename')
    args = parser.parse_args()
    main(args)
