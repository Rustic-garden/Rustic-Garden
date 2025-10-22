import pandas as pd
import numpy as np
import logging
from io import BytesIO
import requests

# -----------------------------------------------------------------------------
# Setup: Logging for GitHub/Production Use
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -----------------------------------------------------------------------------
# Enhanced Reader with Smart Detection + Preprocessing
# -----------------------------------------------------------------------------
def read_table(
    path_or_url: str,
    date_col_hints=('date', 'C0/date', 'Date', 'timestamp', 'datetime'),
    ndvi_hints=('ndvi_mean', 'C0/mean', 'ndvi', 'NDVI', 'C0_mean'),
    dropna=True,
    add_features=True
):
    """
    Read CSV or Excel from local path or URL and normalize columns for time series analysis.
    Supports:
      - OneDrive, GitHub, or HTTP(S) URLs
      - Excel or CSV (auto-detected)
      - Smart date and NDVI column identification
    Returns:
      DataFrame ready for Mann–Kendall, Sen’s slope, PCA, regression, etc.
    """

    logging.info(f"🔍 Loading data from: {path_or_url}")

    # --- Handle remote files (OneDrive, GitHub raw, etc.) ---
    try:
        if str(path_or_url).startswith(("http://", "https://")):
            response = requests.get(path_or_url)
            response.raise_for_status()
            if path_or_url.lower().endswith((".xls", ".xlsx")):
                df = pd.read_excel(BytesIO(response.content))
            else:
                df = pd.read_csv(BytesIO(response.content), sep=None, engine="python")
        else:
            if path_or_url.lower().endswith((".xls", ".xlsx")):
                df = pd.read_excel(path_or_url)
            else:
                df = pd.read_csv(path_or_url, sep=None, engine="python")
    except Exception as e:
        logging.error(f"❌ Failed to load dataset: {e}")
        raise

    # --- Normalize column names ---
    df.columns = [str(c).strip() for c in df.columns]
    logging.info(f"✅ Columns detected: {df.columns.tolist()}")

    # --- Find date column ---
    date_col = next((c for c in df.columns if c in date_col_hints), None)
    if not date_col:
        date_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
    if not date_col:
        raise ValueError("No date column found. Please specify date column name.")

    df['date'] = pd.to_datetime(df[date_col], errors='coerce')

    # --- Find NDVI column ---
    ndvi_col = next((c for c in df.columns if c in ndvi_hints), None)
    if not ndvi_col:
        ndvi_col = next((c for c in df.columns if 'ndvi' in c.lower() or ('mean' in c.lower() and 'c0' in c.lower())), None)
    if not ndvi_col:
        raise ValueError("No NDVI or mean reflectance column found.")

    df['ndvi_mean'] = pd.to_numeric(df[ndvi_col], errors='coerce')

    # --- Optional cleanup ---
    if dropna:
        df = df.dropna(subset=['ndvi_mean', 'date'])

    df = df.sort_values('date').reset_index(drop=True)

    # --- Feature generation for PCA/Regression ---
    if add_features:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'ndvi_mean' not in numeric_cols:
            numeric_cols.append('ndvi_mean')

        # Add lag and rolling average features
        df['ndvi_lag1'] = df['ndvi_mean'].shift(1)
        df['ndvi_3day_avg'] = df['ndvi_mean'].rolling(3, min_periods=1).mean()
        df['ndvi_rate'] = df['ndvi_mean'].diff()

        logging.info(f"🧩 Added derived columns for PCA/regression: {['ndvi_lag1','ndvi_3day_avg','ndvi_rate']}")

    logging.info(f"✅ Final dataset shape: {df.shape}")
    return df

# -----------------------------------------------------------------------------
# Example usage:
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    onedrive_url = "https://1drv.ms/x/c/d10c41df36c93952/EXxfHUfRBJRMqhmZogcfN5gBQya79HTwXLVRCw2HaRGk3w?e=KMvNdV"
    df = read_table(onedrive_url)
    print(df.head())
