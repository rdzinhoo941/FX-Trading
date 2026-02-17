"""
================================================================================
  FX PORTFOLIO OPTIMIZATION ENGINE
  ════════════════════════════════════════════════════════════════════════════
  Wavelet-LSTM · XGBoost · HMM Regime Detection · Hurst Exponent
  Black-Litterman · Markowitz · Full Backtest & Risk Audit

  Pipeline:
    1.  Data Acquisition      — Yahoo Finance (prices) + FRED (rates)
    2.  Data Audit             — Sanity checks, NaN detection, visual QA
    3.  Wavelet Denoising      — db4, VisuShrink (Donoho & Johnstone 1994)
    4.  Feature Engineering    — Classical + Wavelet + Complexity (Shannon, Rényi)
    5.  Hurst Exponent         — R/S method (Peters 1994), fractal dimension
    6.  HMM Regime Detection   — 3-state Gaussian HMM on raw returns
    7.  Wavelet-LSTM           — 64→32 architecture, EarlyStopping
    8.  XGBoost                — Gradient boosting + feature importance
    9.  Ensemble               — 50/50 LSTM+XGB × HMM regime boost
   10.  Black-Litterman        — Momentum + Carry views, posterior optimization
   11.  Markowitz              — Min-variance + Hurst per-pair adjustment
   12.  Backtest               — Weekly rebal, transaction costs, no look-ahead
   13.  Risk Analysis          — Sharpe, Sortino, Calmar, VaR, ES, Max DD
   14.  Visualization          — Dashboard, signal analysis, allocation, perf

  References:
    - Donoho & Johnstone (1994) — VisuShrink wavelet thresholding
    - Peters (1994)             — Hurst exponent via R/S analysis
    - Tang et al. (2023)        — Wavelet-LSTM for financial forecasting
    - Garcin (2023)             — Entropy-based complexity in finance
    - Black & Litterman (1992)  — Bayesian asset allocation
================================================================================
"""

# ==============================================================================
#  IMPORTS
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
from hmmlearn import hmm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pywt

import yfinance as yf
import pandas_datareader.data as web
import datetime
import os
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


# ==============================================================================
#  GLOBAL CONFIGURATION
# ==============================================================================

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
OUTPUT_DIR = "backtest_results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_PRICES  = os.path.join(DATA_DIR, "/Users/thomasbelleville/Downloads/data_forex_prices.csv")
FILE_RATES   = os.path.join(DATA_DIR, "/Users/thomasbelleville/Downloads/data_fred_rates.csv")
FILE_RETURNS = os.path.join(OUTPUT_DIR, "MASTER_RETURNS_BL.csv")

# ── Date Range ───────────────────────────────────────────────────────────────
DOWNLOAD_START = "2010-01-01"
DOWNLOAD_END   = datetime.datetime.now().strftime("%Y-%m-%d")
SYNTH_START    = "2015-01-01"
SYNTH_END      = "2024-12-31"
BACKTEST_START = "2018-01-01"

# ── Forex Pairs ──────────────────────────────────────────────────────────────
PAIRS_YAHOO = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    "EURJPY", "EURGBP", "AUDJPY", "NZDJPY",
    "USDMXN", "USDTRY", "USDZAR", "USDSGD", "USDNOK", "USDSEK",
]
TICKERS_YAHOO = [f"{p}=X" for p in PAIRS_YAHOO]

PAIRS_CORE = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"]

# ── FRED Rate Codes (OECD / Fed) ────────────────────────────────────────────
FRED_MAP = {
    "USD": "DFF",
    "EUR": "IRSTCI01EZM156N",
    "JPY": "IRSTCI01JPM156N",
    "GBP": "IRSTCI01GBM156N",
    "CHF": "IRSTCI01CHM156N",
    "AUD": "IRSTCI01AUM156N",
    "CAD": "IRSTCI01CAM156N",
    "NZD": "IRSTCI01NZM156N",
    "MXN": "IRSTCI01MXM156N",
    "ZAR": "IRSTCI01ZAM156N",
    "TRY": "IRSTCI01TRM156N",
    "NOK": "IRSTCI01NOM156N",
    "SEK": "IRSTCI01SEM156N",
}

# ── Strategy Parameters ──────────────────────────────────────────────────────
TARGET_BUDGET      = 1_000_000.0
TRANSACTION_COST   = 0.0005        # 5 bps per trade
LOOKBACK_MOMENTUM  = 252           # 1 year
LOOKBACK_COV       = 126           # 6 months

# ── Black-Litterman ──────────────────────────────────────────────────────────
RISK_AVERSION_DELTA = 2.5
TAU                 = 0.05
MAX_WEIGHT          = 0.30
GROSS_LEVERAGE      = 1.2

# ── Risk ─────────────────────────────────────────────────────────────────────
CONFIDENCE_LEVEL = 0.95
TRADING_DAYS     = 252

# ── Wavelet ──────────────────────────────────────────────────────────────────
WAVELET_NAME  = "db4"
WAVELET_LEVEL = 4

# ── LSTM ─────────────────────────────────────────────────────────────────────
LSTM_LOOKBACK = 30
LSTM_EPOCHS   = 50
LSTM_BATCH    = 32

# ── Feature Columns (used by LSTM & XGBoost) ────────────────────────────────
FEATURE_COLS = [
    "Return_1d", "Return_5d", "SMA_21", "MACD", "RSI", "RealizedVol_21d",
    "Return_Clean_1d", "Return_Clean_5d", "MACD_Clean",
    "Shannon_Entropy", "Renyi_Entropy",
    "Hurst", "Fractal_Dimension",
]

# ── Visual Style ─────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#FAFAFA",
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   13,
    "axes.titleweight":  "bold",
    "figure.dpi":       120,
})
sns.set_theme(style="whitegrid")

COLORS = {
    "EURUSD": "#1B4F72", "GBPUSD": "#148F77", "USDJPY": "#B9770E",
    "AUDUSD": "#922B21", "USDCAD": "#6C3483", "NZDUSD": "#1A5276",
    "USDCHF": "#7D6608",
    "BULL": "#27AE60", "BEAR": "#E74C3C", "NEUTRAL": "#F39C12",
    "blue": "#2C3E50", "green": "#27AE60", "red": "#E74C3C",
    "purple": "#8E44AD", "gray": "#95A5A6",
}


# ==============================================================================
#  SECTION 0 — PRETTY-PRINTING UTILITIES
# ==============================================================================

def print_header(title: str):
    """Section header with double-line border."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str):
    """Sub-section header."""
    print(f"\n--- {title} ---")


def print_table(headers: list, rows: list):
    """Print an ASCII table with auto-column sizing."""
    if not rows:
        return
    col_widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
        for i, h in enumerate(headers)
    ]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def fmt(vals):
        return "| " + " | ".join(
            str(v).center(w) for v, w in zip(vals, col_widths)
        ) + " |"

    print(sep)
    print(fmt(headers))
    print(sep)
    for row in rows:
        print(fmt(row))
    print(sep)


# ==============================================================================
#  SECTION 1 — DATA ACQUISITION (Yahoo Finance + FRED)
# ==============================================================================

def download_forex_prices() -> pd.DataFrame:
    """
    Download daily Close prices for all FX pairs from Yahoo Finance.
    Forward-fills weekends/holidays, saves to CSV.
    """
    print_header("DATA ACQUISITION — Forex Prices (Yahoo Finance)")
    print(f"  Pairs : {len(TICKERS_YAHOO)}")
    print(f"  Range : {DOWNLOAD_START} → {DOWNLOAD_END}")

    prices = yf.download(
        TICKERS_YAHOO, start=DOWNLOAD_START, end=DOWNLOAD_END, progress=True
    )["Close"]
    prices = prices.ffill().dropna(how="all")
    prices.to_csv(FILE_PRICES)

    print(f"  [OK] Saved → {FILE_PRICES}  ({prices.shape[0]} rows × {prices.shape[1]} cols)")
    return prices


def download_fred_rates() -> pd.DataFrame:
    """
    Download central-bank / overnight rates from FRED (OECD series).
    Resamples monthly series to daily via ffill, converts % → decimal.
    """
    print_header("DATA ACQUISITION — Interest Rates (FRED)")
    rates_df = pd.DataFrame()
    failed = []

    for curr, code in FRED_MAP.items():
        print(f"  ... {curr} ({code})", end="")
        try:
            df = web.DataReader(code, "fred", DOWNLOAD_START, DOWNLOAD_END)
            df.columns = [curr]
            rates_df = df if rates_df.empty else rates_df.join(df, how="outer")
            print("  ✓")
        except Exception as e:
            print(f"  ✗  ({e})")
            failed.append(curr)

    if rates_df.empty:
        print("  [ERROR] No rate data retrieved.")
        return pd.DataFrame()

    # Resample to daily, ffill, convert pct → decimal
    rates_daily = rates_df.resample("D").ffill().ffill() / 100.0
    rates_daily.to_csv(FILE_RATES)

    print(f"\n  [OK] Saved → {FILE_RATES}  ({rates_daily.shape[0]} rows × {rates_daily.shape[1]} cols)")
    print(f"       Format : decimal (0.05 = 5%)")
    if failed:
        print(f"  [WARN] Missing currencies: {failed}")
    return rates_daily


def load_local_data():
    """
    Load pre-downloaded CSVs from disk.
    Returns (prices, rates_daily) aligned on their common date index.
    """
    prices     = pd.read_csv(FILE_PRICES, index_col=0, parse_dates=True)
    rates_daily = pd.read_csv(FILE_RATES,  index_col=0, parse_dates=True)

    common = prices.index.intersection(rates_daily.index)
    prices      = prices.loc[common]
    rates_daily = rates_daily.loc[common]

    return prices, rates_daily


# ==============================================================================
#  SECTION 2 — DATA AUDIT & VISUAL QUALITY CHECK
# ==============================================================================

def run_data_audit(prices: pd.DataFrame, rates: pd.DataFrame):
    """
    Quick sanity check: date ranges, NaN counts, plus 3-panel chart
    (rates evolution, normalised FX prices, carry spread USD-JPY).
    """
    print_header("DATA AUDIT & VISUAL CHECK")
    rates_pct = rates * 100  # for display

    # ── Date ranges ──────────────────────────────────────────────────────────
    print(f"  Prices : {prices.index[0].date()} → {prices.index[-1].date()}  "
          f"({prices.shape[0]} rows, {prices.shape[1]} cols)")
    print(f"  Rates  : {rates.index[0].date()} → {rates.index[-1].date()}  "
          f"({rates.shape[0]} rows, {rates.shape[1]} cols)")

    # ── Missing data (last 100 days) ────────────────────────────────────────
    miss_p = prices.iloc[-100:].isna().sum().sum()
    miss_r = rates.iloc[-100:].isna().sum().sum()
    print(f"\n  NaN (last 100 days) — Prices: {miss_p}  |  Rates: {miss_r}")

    # ── 3-panel chart ────────────────────────────────────────────────────────
    print("\n  Generating audit charts...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

    # Panel 1: interest rates
    major = [c for c in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"] if c in rates_pct.columns]
    if major:
        rates_pct[major].plot(ax=axes[0], linewidth=1.5)
        axes[0].set_title("1. Central Bank Rates (FRED / OECD)", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Rate (%)")
        axes[0].legend(loc="upper left", ncol=2)
        axes[0].grid(True, alpha=0.3)

    # Panel 2: normalised FX prices (base 100)
    fx_show = [p for p in ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"] if p in prices.columns]
    if fx_show:
        norm = (prices[fx_show] / prices[fx_show].iloc[0]) * 100
        norm.plot(ax=axes[1], linewidth=1.5)
        axes[1].set_title("2. Relative FX Performance (Base 100)", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Base 100")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)

    # Panel 3: carry spread USD − JPY
    if "USD" in rates_pct.columns and "JPY" in rates_pct.columns:
        spread = rates_pct["USD"] - rates_pct["JPY"]
        axes[2].plot(spread.index, spread.values, color="darkgreen", linewidth=1.5, label="Spread USD − JPY")
        axes[2].fill_between(spread.index, 0, spread, where=(spread >= 0),
                             color="green", alpha=0.3, interpolate=True)
        axes[2].fill_between(spread.index, 0, spread, where=(spread < 0),
                             color="red", alpha=0.3, interpolate=True)
        axes[2].set_title("3. Carry Trade Driver: USD − JPY Rate Differential (%)",
                          fontsize=14, fontweight="bold")
        axes[2].set_ylabel("Differential (%)")
        axes[2].legend(loc="upper left")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig0_data_audit.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("  [OK] fig0_data_audit.png")


# ==============================================================================
#  SECTION 3 — SYNTHETIC DATA GENERATION (fallback / demo mode)
# ==============================================================================

def load_real_data_for_pipeline() -> dict:
    """
    Load real Yahoo FX prices from CSV and build the data_dict
    used by the ML pipeline (wavelet, features, LSTM, XGBoost, etc.)
    """
    print_header("LOADING REAL DATA FOR ML PIPELINE")

    prices = pd.read_csv(FILE_PRICES, index_col=0, parse_dates=True)
    prices = prices.loc["2015-01-01":]
    prices = prices.ffill().dropna(how="all")

    # Mapping: nom interne → colonne Yahoo
    pair_map = {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
        "AUDUSD": "AUDUSD=X",
        "USDCAD": "USDCAD=X",
        "NZDUSD": "NZDUSD=X",
        "USDCHF": "USDCHF=X",
    }

    data_dict = {}
    for pair, ticker in pair_map.items():
        if ticker not in prices.columns:
            print(f"  [WARN] {ticker} not found in CSV — skipping")
            continue

        close = prices[ticker].dropna()
        df = pd.DataFrame({
            "Open":  close.shift(1),       # Approximation: open = close veille
            "High":  close * 1.002,         # Approximation simple
            "Low":   close * 0.998,
            "Close": close,
        }, index=close.index)
        df.dropna(inplace=True)
        data_dict[pair] = df
        print(f"  [OK] {pair} — {len(df)} days ({df.index[0].date()} → {df.index[-1].date()})")

    print(f"\n  Loaded {len(data_dict)} pairs with real market data")
    return data_dict


# ==============================================================================
#  SECTION 4 — WAVELET DENOISING
#  Reference: Donoho & Johnstone (1994) — VisuShrink
#             Tang et al. (2023)        — application to FX
# ==============================================================================

def wavelet_denoise(signal: np.ndarray, wavelet: str = WAVELET_NAME,
                    level: int = WAVELET_LEVEL, threshold_mode: str = "soft"):
    """
    Denoise a 1-D signal via Discrete Wavelet Transform.
    Pipeline: Signal → DWT → VisuShrink thresholding → IDWT → Clean signal

    VisuShrink threshold: σ * √(2 · log N)
    where σ = MAD(detail coefficients at finest level) / 0.6745
    """
    signal = np.asarray(signal, dtype=float)
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Robust noise estimate from finest detail coefficients
    sigma     = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Threshold detail coefficients, keep approximation intact
    coeffs_d    = list(coeffs)
    coeffs_d[0] = coeffs[0]
    for i in range(1, len(coeffs)):
        coeffs_d[i] = pywt.threshold(coeffs[i], threshold, mode=threshold_mode)

    clean = pywt.waverec(coeffs_d, wavelet)

    # Ensure output length matches input
    if len(clean) > len(signal):
        clean = clean[: len(signal)]
    elif len(clean) < len(signal):
        clean = np.pad(clean, (0, len(signal) - len(clean)), mode="edge")

    return clean, coeffs


def apply_wavelet_denoising(data_dict: dict, pairs: list) -> list:
    """
    Apply wavelet denoising to each pair's Close, store Close_Clean
    and Return_Clean columns.  Returns summary stats for reporting.
    """
    print_header("WAVELET DENOISING (db4, level=4, VisuShrink)")
    stats = []

    for pair in pairs:
        df = data_dict[pair]
        raw   = df["Close"].values.copy()
        clean, _ = wavelet_denoise(raw)

        df["Close_Clean"]  = clean
        df["Return_Clean"] = pd.Series(clean, index=df.index).pct_change()

        residual   = raw - clean
        snr_before = np.mean(raw ** 2) / np.var(raw - np.mean(raw))
        snr_after  = (np.mean(clean ** 2) / np.var(residual)) if np.var(residual) > 0 else np.inf
        noise_pct  = np.std(residual) / np.std(raw) * 100

        stats.append([pair, f"{noise_pct:.1f}%", f"{snr_before:.1f}", f"{snr_after:.1f}"])

    print_table(["Pair", "Noise removed", "SNR before", "SNR after"], stats)
    return stats


# ==============================================================================
#  SECTION 5 — FEATURE ENGINEERING
#  Classical + Wavelet-clean + Complexity (Shannon, Rényi) — Garcin (2023)
# ==============================================================================

def calculate_shannon_entropy(returns: np.ndarray, bins: int = 50) -> float:
    """Shannon entropy: S = −Σ pᵢ · log(pᵢ) · dx"""
    hist, _ = np.histogram(returns, bins=bins, density=True)
    hist = hist[hist > 0]
    dx = (returns.max() - returns.min()) / bins
    return -np.sum(hist * np.log(hist) * dx)


def calculate_renyi_entropy(returns: np.ndarray, alpha: float = 2,
                            bins: int = 50) -> float:
    """Rényi entropy: Hα = 1/(1−α) · log(Σ pᵢ^α)"""
    hist, _ = np.histogram(returns, bins=bins, density=True)
    hist = hist[hist > 0]
    dx = (returns.max() - returns.min()) / bins
    return (1 / (1 - alpha)) * np.log(np.sum((hist * dx) ** alpha))


def rolling_entropy(series: pd.Series, window: int = 100,
                    entropy_type: str = "shannon") -> pd.Series:
    """Compute rolling Shannon or Rényi entropy."""
    func = calculate_shannon_entropy if entropy_type == "shannon" else calculate_renyi_entropy
    values = []
    for i in range(window, len(series)):
        subset = series.iloc[i - window: i].dropna().values
        if len(subset) > 10:
            try:
                values.append(func(subset))
            except Exception:
                values.append(np.nan)
        else:
            values.append(np.nan)
    return pd.Series([np.nan] * window + values, index=series.index)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature set for one pair:
      • 6 classical  : Return_1d/5d, SMA_21, MACD, RSI, RealizedVol_21d
      • 3 wavelet    : Return_Clean_1d/5d, MACD_Clean
      • 2 complexity : Shannon_Entropy, Renyi_Entropy
      • 1 target     : Target_Direction (next-day direction)
    Hurst & Fractal_Dimension are added separately (section 6).
    """
    df = df.copy()

    # ── Classical ────────────────────────────────────────────────────────────
    df["Return"]         = df["Close"].pct_change()
    df["Return_1d"]      = df["Return"].shift(1)
    df["Return_5d"]      = df["Close"].pct_change(5)
    df["SMA_21"]         = df["Close"].rolling(21).mean()
    df["EMA_12"]         = df["Close"].ewm(span=12).mean()
    df["EMA_26"]         = df["Close"].ewm(span=26).mean()
    df["MACD"]           = df["EMA_12"] - df["EMA_26"]

    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / loss
    df["RSI"]            = 100 - (100 / (1 + rs))
    df["RealizedVol_21d"] = df["Return"].rolling(21).std() * np.sqrt(252)

    # ── Wavelet-clean ────────────────────────────────────────────────────────
    df["Return_Clean_1d"] = df["Return_Clean"].shift(1)
    df["Return_Clean_5d"] = df["Close_Clean"].pct_change(5)
    df["SMA_Clean_21"]    = df["Close_Clean"].rolling(21).mean()
    df["MACD_Clean"]      = (df["Close_Clean"].ewm(span=12).mean()
                             - df["Close_Clean"].ewm(span=26).mean())

    # ── Complexity — Garcin (2023) ───────────────────────────────────────────
    df["Shannon_Entropy"] = rolling_entropy(df["Return"], 100, "shannon")
    df["Renyi_Entropy"]   = rolling_entropy(df["Return"], 100, "renyi")

    # ── Target ───────────────────────────────────────────────────────────────
    df["Target_Direction"] = (df["Return"].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    return df


def apply_feature_engineering(data_dict: dict, pairs: list):
    """Apply create_features() to every pair and print summary."""
    print_header("FEATURE ENGINEERING")
    for pair in pairs:
        data_dict[pair] = create_features(data_dict[pair])
    print(f"  Classical features :  6  (Return_1d/5d, SMA_21, MACD, RSI, RealizedVol)")
    print(f"  Wavelet features   :  3  (Return_Clean_1d/5d, MACD_Clean)")
    print(f"  Complexity features:  2  (Shannon_Entropy, Renyi_Entropy)")
    print(f"  Hurst features     :  2  (Hurst, Fractal_Dimension)  [added next]")
    print(f"  Total              : 13 features per pair")


# ==============================================================================
#  SECTION 6 — HURST EXPONENT (R/S method — Peters 1994)
# ==============================================================================

def calculate_hurst_exponent(ts: np.ndarray, max_lag: int = 20) -> float:
    """
    Estimate the Hurst exponent via Rescaled Range (R/S) analysis.
      H > 0.5  → persistent (trending)
      H ≈ 0.5  → Brownian motion (random walk)
      H < 0.5  → anti-persistent (mean-reverting)
    """
    n = len(ts)
    if n < 20:
        return 0.5

    lags = range(2, min(max_lag, n // 2))
    tau = []

    for lag in lags:
        n_chunks = n // lag
        if n_chunks == 0:
            continue
        rs_values = []
        for i in range(n_chunks):
            chunk = ts[i * lag: (i + 1) * lag]
            if len(chunk) < 2:
                continue
            mean_c = np.mean(chunk)
            cumsum_dev = np.cumsum(chunk - mean_c)
            R = np.max(cumsum_dev) - np.min(cumsum_dev)
            S = np.std(chunk)
            if S > 0:
                rs_values.append(R / S)
        if rs_values:
            tau.append(np.mean(rs_values))

    if len(tau) < 2:
        return 0.5

    lags_arr = np.array(list(lags)[: len(tau)])
    tau_arr  = np.array(tau)
    if np.any(tau_arr <= 0):
        return 0.5

    slope = np.polyfit(np.log(lags_arr), np.log(tau_arr), 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


def rolling_hurst(series: pd.Series, window: int = 100) -> pd.Series:
    """Compute rolling Hurst exponent over a sliding window."""
    values = []
    for i in range(window, len(series)):
        try:
            values.append(calculate_hurst_exponent(series.iloc[i - window: i].values))
        except Exception:
            values.append(np.nan)
    return pd.Series([np.nan] * window + values, index=series.index)


def apply_hurst(data_dict: dict, pairs: list) -> list:
    print_header("HURST EXPONENT (R/S method, window=100)")
    table = []
    for pair in pairs:
        df = data_dict[pair]
        # CORRECTION : Hurst sur les rendements, pas les prix
        df["Hurst"]             = rolling_hurst(df["Return"].dropna(), 100)
        df["Fractal_Dimension"] = 2 - df["Hurst"]
        df.dropna(inplace=True)

        h_mean = df["Hurst"].mean()
        h_last = df["Hurst"].iloc[-1]
        regime = ("TRENDING" if h_mean > 0.55
                  else ("MEAN-REV" if h_mean < 0.45 else "BROWNIAN"))
        table.append([pair, f"{h_mean:.4f}", f"{h_last:.4f}", regime])

    print_table(["Pair", "H mean", "H current", "Regime"], table)
    return table


# ==============================================================================
#  SECTION 7 — HMM REGIME DETECTION (3-state Gaussian)
#  Uses RAW returns to avoid over-smoothing from the wavelet step.
# ==============================================================================

def apply_hmm_regimes(data_dict: dict, pairs: list):
    """
    Fit a 3-state Gaussian HMM to each pair's raw returns.
    Labels states as BULL / NEUTRAL / BEAR based on mean return.
    Returns (hmm_regimes, hmm_details) dicts.
    """
    print_header("HMM REGIME DETECTION (3 states, raw returns)")
    hmm_regimes = {}
    hmm_details = {}

    for pair in pairs:
        df      = data_dict[pair]
        returns = df["Return"].values.reshape(-1, 1)
        mask    = ~np.isnan(returns).flatten() & ~np.isinf(returns).flatten()
        r_clean = returns[mask]

        model = hmm.GaussianHMM(
        n_components=3, covariance_type="diag", n_iter=300, tol=1e-4, random_state=42)
        model.fit(r_clean)
        states = model.predict(r_clean)

        states_full = np.full(len(df), -1)
        states_full[mask] = states
        df["HMM_Regime"] = states_full

        # Rank states by mean return
        means = []
        for s in range(3):
            r = df.loc[states_full == s, "Return"]
            means.append(r.mean() if len(r) > 0 else 0)
        order = np.argsort(means)
        label_map = {order[0]: "BEAR", order[1]: "NEUTRAL", order[2]: "BULL", -1: "NEUTRAL"}
        df["HMM_Regime_Label"] = df["HMM_Regime"].map(label_map)

        current = df["HMM_Regime_Label"].iloc[-1]
        hmm_regimes[pair] = current

        vols = []
        for lbl in ["BULL", "NEUTRAL", "BEAR"]:
            r = df.loc[df["HMM_Regime_Label"] == lbl, "Return"]
            vols.append(r.std() * np.sqrt(252) if len(r) > 0 else 0)

        hmm_details[pair] = {
            "means":   [means[order[2]], means[order[1]], means[order[0]]],
            "vols":    vols,
            "current": current,
        }

    # Print summary
    rows = []
    for pair in pairs:
        d = hmm_details[pair]
        rows.append([
            pair, d["current"],
            f"{d['means'][0]*100:+.3f}%", f"{d['means'][2]*100:+.3f}%",
            f"{d['vols'][0]*100:.1f}%",   f"{d['vols'][2]*100:.1f}%",
        ])
    print_table(["Pair", "Regime", "μ BULL", "μ BEAR", "σ BULL", "σ BEAR"], rows)

    return hmm_regimes, hmm_details


# ==============================================================================
#  SECTION 8 — WAVELET-LSTM (64→32, EarlyStopping)
#  Reference: Tang et al. (2023)
# ==============================================================================

def _create_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    """Reshape flat arrays into (samples, lookback, features) sequences."""
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i: i + lookback])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)


def train_lstm_models(data_dict: dict, pairs: list):
    """
    Train one Wavelet-LSTM per pair (binary direction classifier).
    Architecture: LSTM(64) → Dropout(0.3) → LSTM(32) → Dropout(0.3)
                  → Dense(16, relu) → Dense(1, sigmoid)
    Returns (predictions, accuracies) dicts.
    """
    print_header("WAVELET-LSTM (64→32, EarlyStopping)")
    predictions = {}
    accuracies  = {}

    for pair in pairs:
        df = data_dict[pair]
        X  = df[FEATURE_COLS].values
        y  = df["Target_Direction"].values

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_train)
        X_te_s   = scaler.transform(X_test)

        X_tr_seq, y_tr_seq = _create_sequences(X_tr_s, y_train, LSTM_LOOKBACK)
        X_te_seq, y_te_seq = _create_sequences(X_te_s, y_test,  LSTM_LOOKBACK)

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(LSTM_LOOKBACK, X_train.shape[1])),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1,  activation="sigmoid"),
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)  # était patience=3
        model.fit(
            X_tr_seq, y_tr_seq,
            epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
            validation_split=0.1, verbose=0, callbacks=[es],
        )

        y_prob = model.predict(X_te_seq, verbose=0).flatten()
        acc    = np.mean((y_prob > 0.5) == y_te_seq)
        e_ret  = (y_prob.mean() - 0.5) * 2 * df["RealizedVol_21d"].mean() / np.sqrt(252)

        predictions[pair] = e_ret
        accuracies[pair]  = acc

    rows = [[p, f"{predictions[p]*10000:+.2f} bps", f"{accuracies[p]*100:.1f}%"] for p in pairs]
    print_table(["Pair", "E[r] predicted", "Accuracy"], rows)
    return predictions, accuracies


# ==============================================================================
#  SECTION 9 — XGBOOST + FEATURE IMPORTANCE
# ==============================================================================

def train_xgboost_models(data_dict: dict, pairs: list):
    """
    Train one XGBClassifier per pair (binary direction).
    Returns (predictions, accuracies, feature_importances) dicts.
    """
    print_header("XGBOOST + FEATURE IMPORTANCE")
    predictions  = {}
    accuracies   = {}
    importances  = {}

    for pair in pairs:
        df = data_dict[pair]
        X  = df[FEATURE_COLS]
        y  = df["Target_Direction"]

        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        mdl = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            random_state=42, use_label_encoder=False, eval_metric="logloss",
        )
        mdl.fit(X_tr_s, y_train)

        y_prob = mdl.predict_proba(X_te_s)
        acc    = np.mean(mdl.predict(X_te_s) == y_test)
        e_ret  = (y_prob[:, 1].mean() - 0.5) * 2 * df["RealizedVol_21d"].mean() / np.sqrt(252)

        predictions[pair] = e_ret
        accuracies[pair]  = acc
        importances[pair] = dict(zip(FEATURE_COLS, mdl.feature_importances_))

    rows = [[p, f"{predictions[p]*10000:+.2f} bps", f"{accuracies[p]*100:.1f}%"] for p in pairs]
    print_table(["Pair", "E[r] predicted", "Accuracy"], rows)

    # Average feature importance
    avg = {f: np.mean([importances[p][f] for p in pairs]) for f in FEATURE_COLS}
    sorted_fi = sorted(avg.items(), key=lambda x: x[1], reverse=True)

    print_subheader("Average Feature Importance (XGBoost)")
    fi_rows = [[f, f"{v:.4f}", "█" * int(v * 50)] for f, v in sorted_fi]
    print_table(["Feature", "Importance", ""], fi_rows)

    return predictions, accuracies, importances, sorted_fi


# ==============================================================================
#  SECTION 10 — ENSEMBLE (LSTM + XGBoost × HMM boost)
#  Reference: Liu et al. — Ensemble methods for financial prediction
# ==============================================================================

def build_ensemble(pairs: list, lstm_preds: dict, xgb_preds: dict,
                   hmm_regimes: dict):
    print_header("ENSEMBLE (0.5×LSTM + 0.5×XGB) × HMM boost")
    boost_map = {"BULL": 1.2, "BEAR": 0.8, "NEUTRAL": 1.0}
    ensemble = {}
    rows = []
    MIN_SIGNAL = TRANSACTION_COST * 0.5  # signal doit couvrir au moins 50% du coût = 2.5 bps  # 5 bps = coût de transaction

    for pair in pairs:
        lp = lstm_preds[pair]
        xp = xgb_preds[pair]
        regime = hmm_regimes[pair]
        b  = boost_map[regime]
        ep = (0.5 * lp + 0.5 * xp) * b

        if abs(ep) < MIN_SIGNAL:
            ep = 0.0

        ensemble[pair] = ep
        rows.append([
            pair, f"{lp*10000:+.2f}", f"{xp*10000:+.2f}",
            regime, f"×{b}", f"{ep*10000:+.2f}",
        ])

    print_table(["Pair", "LSTM(bps)", "XGB(bps)", "HMM", "Boost", "Ensemble(bps)"], rows)
    return ensemble

# ==============================================================================
#  SECTION 11 — MARKOWITZ OPTIMIZATION
#  min w^T Σ w   s.t.  w^T 1 = 1,  w^T μ ≥ r_target
# ==============================================================================

def markowitz_optimization(mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Minimum-variance optimisation with a return constraint.
    Bounds per asset: [5%, 50%].  Fully invested (sum = 1).
    """
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0.05, 0.50)] * n
    target = np.mean(mu[mu > 0]) if np.any(mu > 0) else np.mean(mu)

    constraints = [
        {"type": "eq",   "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: w @ mu - target},
    ]
    res = minimize(
        lambda w: w @ Sigma @ w,
        w0, method="SLSQP", bounds=bounds, constraints=constraints,
    )
    return res.x


def run_markowitz(data_dict: dict, pairs: list, ensemble_preds: dict):
    """
    Build the covariance matrix from historical returns, solve Markowitz,
    and print allocation + portfolio stats.
    Returns (w_markowitz, Sigma, mu, returns_matrix).
    """
    print_header("MARKOWITZ OPTIMIZATION")

    returns_matrix = pd.DataFrame(
        {pair: data_dict[pair]["Return"] for pair in pairs}
    ).dropna()
    Sigma = returns_matrix.cov().values
    mu    = np.array([ensemble_preds[pair] for pair in pairs])

    w = markowitz_optimization(mu, Sigma)
    port_ret = (w @ mu) * 252
    port_vol = np.sqrt(w @ Sigma @ w) * np.sqrt(252)

    rows = [[pair, f"{w[i]*100:.1f}%", f"{mu[i]*10000:+.2f} bps"]
            for i, pair in enumerate(pairs)]
    rows.append(["TOTAL", "100.0%", f"{np.sum(mu * w)*10000:+.2f} bps"])
    print_table(["Pair", "Weight", "E[r] ensemble"], rows)

    print(f"\n  Annualised return  : {port_ret*100:+.4f}%")
    print(f"  Annualised vol     : {port_vol*100:.4f}%")

    return w, Sigma, mu, returns_matrix


# ==============================================================================
#  SECTION 12 — HURST-BASED WEIGHT ADJUSTMENT
#  f(Hᵢ) = 1 + 2·|Hᵢ − 0.5|
#  → H far from 0.5 = more predictable → overweight
#  → H ≈ 0.5 = Brownian noise → underweight
# ==============================================================================

def adjust_weights_hurst(data_dict: dict, pairs: list,
                         w_markowitz: np.ndarray) -> np.ndarray:
    """
    Multiply each Markowitz weight by a Hurst-derived factor,
    then re-normalise to sum = 1.
    """
    print_header("HURST-BASED WEIGHT ADJUSTMENT")
    factors = {}
    rows = []

    for pair in pairs:
        h = data_dict[pair]["Hurst"].iloc[-1]
        d = abs(h - 0.5)
        f = 1.0 + 2.0 * d
        factors[pair] = f
        regime = "TRENDING" if h > 0.55 else ("MEAN-REV" if h < 0.45 else "BROWNIAN")
        rows.append([pair, f"{h:.4f}", f"|{d:.4f}|", regime, f"×{f:.2f}"])

    print_table(["Pair", "H current", "|H−0.5|", "Regime", "Factor"], rows)

    f_arr  = np.array([factors[p] for p in pairs])
    w_adj  = w_markowitz * f_arr
    w_adj /= w_adj.sum()

    print_subheader("Comparison: Markowitz vs Hurst-Adjusted")
    comp = []
    for i, pair in enumerate(pairs):
        delta = (w_adj[i] - w_markowitz[i]) * 100
        arrow = "▲" if delta > 0.1 else ("▼" if delta < -0.1 else "─")
        comp.append([
            pair, f"{w_markowitz[i]*100:.1f}%", f"{w_adj[i]*100:.1f}%",
            f"{delta:+.1f}pp", arrow,
        ])
    print_table(["Pair", "Markowitz", "Hurst-Adj", "Delta", ""], comp)

    return w_adj


# ==============================================================================
#  SECTION 13 — PERFORMANCE METRICS
# ==============================================================================

def calculate_sharpe(returns: np.ndarray) -> float:
    mu = np.mean(returns)
    sigma = np.std(returns)
    return mu / sigma * np.sqrt(252) if sigma > 0 else 0.0


def calculate_sortino(returns: np.ndarray) -> float:
    downside = returns[returns < 0]
    if len(downside) == 0:
        return np.inf
    d_std = np.sqrt(np.mean(downside ** 2))
    return np.mean(returns) / d_std * np.sqrt(252) if d_std > 0 else 0.0


def calculate_max_drawdown(returns: np.ndarray) -> float:
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    return float(np.min((cum - peak) / peak))


def calculate_calmar(returns: np.ndarray) -> float:
    ann_ret = np.mean(returns) * 252
    mdd = abs(calculate_max_drawdown(returns))
    return ann_ret / mdd if mdd > 0 else 0.0


def calculate_var(returns: np.ndarray, alpha: float = 0.95) -> float:
    return -float(np.percentile(returns, (1 - alpha) * 100))


def calculate_expected_shortfall(returns: np.ndarray,
                                  alpha: float = 0.95) -> float:
    var = calculate_var(returns, alpha)
    tail = -returns
    tail = tail[tail >= var]
    return float(np.mean(tail)) if len(tail) > 0 else var


def print_performance_metrics(portfolios: dict, data_dict: dict,
                              pairs: list, returns_matrix: pd.DataFrame):
    """Print full performance dashboard including per-pair stats."""
    print_header("PERFORMANCE METRICS")

    # ── Portfolio-level metrics ──────────────────────────────────────────────
    rows = []
    for name, rets in portfolios.items():
        rows.append([
            name,
            f"{calculate_sharpe(rets):.4f}",
            f"{calculate_sortino(rets):.4f}",
            f"{calculate_calmar(rets):.4f}",
            f"{calculate_max_drawdown(rets)*100:.2f}%",
            f"{calculate_var(rets)*100:.3f}%",
            f"{calculate_expected_shortfall(rets)*100:.3f}%",
            f"{np.mean(rets > 0)*100:.1f}%",
        ])
    print_table(
        ["Portfolio", "Sharpe", "Sortino", "Calmar", "Max DD",
         "VaR 95%", "ES 95%", "Win Rate"],
        rows,
    )

    # ── Per-pair stats ───────────────────────────────────────────────────────
    print_subheader("Per-pair statistics (daily returns)")
    ps = []
    for pair in pairs:
        r = data_dict[pair]["Return"]
        ps.append([
            pair,
            f"{r.mean()*252*100:+.3f}%",
            f"{r.std()*np.sqrt(252)*100:.2f}%",
            f"{r.skew():.3f}",
            f"{r.kurtosis():.3f}",
        ])
    print_table(["Pair", "Ann. Return", "Ann. Vol", "Skewness", "Kurtosis"], ps)

    # ── Correlation matrix ───────────────────────────────────────────────────
    print_subheader("Correlation Matrix")
    corr = returns_matrix.corr()
    print(corr.round(3).to_string())
    return corr


# ==============================================================================
#  SECTION 14 — BLACK-LITTERMAN ENGINE (Momentum + Carry views)
# ==============================================================================

def parse_pair(ticker: str):
    """'EURUSD=X' → ('EUR', 'USD')"""
    clean = ticker.replace("=X", "")
    return clean[:3], clean[3:]


def compute_total_returns_and_momentum(prices: pd.DataFrame,
                                       rates_daily: pd.DataFrame):
    """
    For each FX pair compute:
      1. Price return  (pct_change)
      2. Carry return  (base rate − quote rate) / 252
      3. Total return  = price + carry
      4. 12-month momentum (sum of log total returns, shifted by 1 day)
    Returns (returns_total, momentum_score) aligned to BACKTEST_START.
    """
    print_subheader("Total Returns (Price + Carry) & Momentum")
    tickers = prices.columns.tolist()
    returns_total  = pd.DataFrame(index=prices.index, columns=tickers)
    momentum_score = pd.DataFrame(index=prices.index, columns=tickers)

    for t in tickers:
        base, quote = parse_pair(t)
        r_price = prices[t].pct_change()

        r_carry = 0.0
        if base in rates_daily.columns and quote in rates_daily.columns:
            r_carry = (rates_daily[base] - rates_daily[quote]) / 252.0

        returns_total[t] = r_price + r_carry

        log_rets = np.log(1 + returns_total[t])
        momentum_score[t] = log_rets.rolling(LOOKBACK_MOMENTUM).sum().shift(1)

    returns_total.dropna(inplace=True)
    momentum_score.dropna(inplace=True)

    returns_total  = returns_total.loc[BACKTEST_START:]
    momentum_score = momentum_score.loc[BACKTEST_START:]
    prices_bt      = prices.loc[BACKTEST_START:]

    print(f"  Backtest window: {returns_total.index[0].date()} → {returns_total.index[-1].date()}")
    return returns_total, momentum_score, prices_bt


def get_black_litterman_weights(curr_date, returns_total: pd.DataFrame,
                                momentum_score: pd.DataFrame) -> np.ndarray:
    """
    Compute Black-Litterman posterior weights for a given date.
    Views are derived from momentum scores (top 4 long, bottom 4 short).
    """
    tickers_list = returns_total.columns.tolist()
    n = len(tickers_list)

    # Historical window for covariance
    start = curr_date - pd.Timedelta(days=int(LOOKBACK_COV * 1.5))
    history = returns_total.loc[start: curr_date].tail(LOOKBACK_COV)
    if len(history) < LOOKBACK_COV * 0.9:
        return np.zeros(n)

    # 1. Prior (equilibrium)
    sigma = history.cov() * 252
    w_mkt = np.ones(n) / n
    pi    = RISK_AVERSION_DELTA * sigma.dot(w_mkt)

    # 2. Views from momentum
    if curr_date not in momentum_score.index:
        return np.zeros(n)
    curr_mom = momentum_score.loc[curr_date]
    ranked   = curr_mom.sort_values(ascending=False)
    views_assets = ranked.head(4).index.tolist() + ranked.tail(4).index.tolist()

    P, Q, Omega_list = [], [], []
    for t in views_assets:
        idx = tickers_list.index(t)
        row = np.zeros(n); row[idx] = 1
        P.append(row)
        Q.append(np.tanh(curr_mom[t]) * 0.20)
        Omega_list.append(sigma.iloc[idx, idx] * TAU)

    P     = np.array(P)
    Q     = np.array(Q)
    Omega = np.diag(Omega_list)

    # 3. Posterior (BL master formula)
    try:
        tau_sigma     = TAU * sigma
        inv_tau_sigma = np.linalg.inv(tau_sigma + np.eye(n) * 1e-6)
        inv_omega     = np.linalg.inv(Omega)
        M_inv         = inv_tau_sigma + P.T @ inv_omega @ P
        M             = np.linalg.inv(M_inv + np.eye(n) * 1e-6)
        bl_returns    = M @ (inv_tau_sigma @ pi + P.T @ inv_omega @ Q)
    except np.linalg.LinAlgError:
        return np.zeros(n)

    # 4. Mean-variance optimisation
    def neg_utility(w):
        ret = w @ bl_returns
        vol = np.sqrt(w @ sigma.values @ w)
        return -(ret - (RISK_AVERSION_DELTA / 2) * vol ** 2)

    cons   = ({"type": "ineq", "fun": lambda x: GROSS_LEVERAGE - np.sum(np.abs(x))},)
    bounds = tuple((-MAX_WEIGHT, MAX_WEIGHT) for _ in range(n))
    init   = np.sign(bl_returns) * (1 / n)

    try:
        res = minimize(neg_utility, init, method="SLSQP", bounds=bounds,
                       constraints=cons, tol=1e-6)
        return res.x
    except Exception:
        return np.zeros(n)


# ==============================================================================
#  SECTION 15 — BACKTEST ENGINE (NO LOOK-AHEAD BIAS)
# ==============================================================================

def run_backtest(returns_total: pd.DataFrame,
                 momentum_score: pd.DataFrame):
    """
    Walk-forward backtest with weekly Friday rebalancing.
    Weights are computed at Friday close and applied starting Monday.
    Transaction costs are deducted on each rebalance.
    Returns (equity_curve, equity_dates, strategy_returns, capital).
    """
    print_header("BACKTEST ENGINE (strict no-lookahead)")

    tickers     = returns_total.columns.tolist()
    rebal_dates = set(returns_total.resample("W-FRI").last().index)
    daily_dates = returns_total.index

    current_w = np.zeros(len(tickers))
    capital   = TARGET_BUDGET
    equity    = [capital]
    eq_dates  = [daily_dates[0]]
    strat_ret = []
    last_month = 0

    print("-" * 100)
    print(f"{'DATE':<12} | {'EQUITY ($)':<14} | ALLOCATION (effective next period)")
    print("-" * 100)

    for d in daily_dates[1:]:
        # 1. Apply yesterday's weights to today's returns
        day_ret  = returns_total.loc[d]
        port_ret = current_w @ day_ret
        capital *= (1 + port_ret)

        # 2. Rebalance at Friday close
        cost = 0.0
        if d in rebal_dates:
            target_w = get_black_litterman_weights(d, returns_total, momentum_score)
            target_w[np.abs(target_w) < 0.02] = 0

            turnover = np.sum(np.abs(target_w - current_w))
            cost     = turnover * TRANSACTION_COST
            capital -= capital * cost

            current_w = target_w

            # Monthly log
            if d.month != last_month:
                longs  = [f"{tickers[j].replace('=X','')}: {target_w[j]:.0%}"
                          for j in range(len(target_w)) if target_w[j] > 0.05]
                shorts = [f"{tickers[j].replace('=X','')}: {target_w[j]:.0%}"
                          for j in range(len(target_w)) if target_w[j] < -0.05]
                print(f"{d.date()} | {capital:>13,.0f} | "
                      f"L: [{', '.join(longs) or 'None'}]  "
                      f"S: [{', '.join(shorts) or 'None'}]")
                last_month = d.month

        equity.append(capital)
        eq_dates.append(d)
        strat_ret.append(port_ret - (cost if d in rebal_dates else 0.0))

    print("-" * 100)
    print(f"  Final Capital: {capital:,.0f} $")

    # Export
    df_export = returns_total.loc[daily_dates[1:]].copy()
    df_export["STRATEGY"] = strat_ret
    df_export.to_csv(FILE_RETURNS)

    return equity, eq_dates, strat_ret, capital


# ==============================================================================
#  SECTION 16 — BL RISK ANALYSIS
# ==============================================================================

def run_bl_risk_analysis(capital: float):
    """Post-backtest risk audit on the BL strategy."""
    print_header("BLACK-LITTERMAN — RISK AUDIT")
    df = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    sr = df["STRATEGY"]

    vol  = sr.std() * np.sqrt(TRADING_DAYS)
    ret  = sr.mean() * TRADING_DAYS
    shp  = ret / vol if vol > 0 else 0
    nav  = (1 + sr).cumprod()
    dd   = (nav - nav.cummax()) / nav.cummax()
    mdd  = dd.min()
    calm = ret / abs(mdd) if mdd < 0 else 0
    var  = sr.quantile(1 - CONFIDENCE_LEVEL)

    print(f"  Ann. Return    : {ret:+.2%}")
    print(f"  Total Return   : {(capital / TARGET_BUDGET) - 1:+.2%}")
    print(f"  Sharpe         : {shp:.2f}")
    print(f"  Calmar         : {calm:.2f}")
    print(f"  Max Drawdown   : {mdd:.2%}")
    print(f"  Ann. Volatility: {vol:.2%}")
    print(f"  Daily VaR(95%) : {var:.2%}")

    pd.DataFrame({
        "Metric": ["Ann. Return", "Sharpe", "Max DD", "Volatility"],
        "Value":  [ret, shp, mdd, vol],
    }).to_csv(os.path.join(OUTPUT_DIR, "RISK_REPORT.csv"), index=False)


# ==============================================================================
#  SECTION 17 — VISUALIZATIONS
# ==============================================================================

def plot_dashboard(portfolios: dict, pairs: list,
                   w_equal: np.ndarray, w_markowitz: np.ndarray,
                   w_hurst: np.ndarray, returns_matrix: pd.DataFrame,
                   corr_matrix: pd.DataFrame):
    """Figure 1: 2×2 dashboard — equity curves, allocation bars, drawdown, corr."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("FX Portfolio Optimization — Dashboard",
                 fontsize=16, fontweight="bold", y=0.98)

    # (0,0) Equity curves
    ax = axes[0, 0]
    style = {
        "1/N (benchmark)": ("#95A5A6", "--", 1.5),
        "Markowitz ML":    (COLORS["blue"], "-", 2),
        "Hurst-Adjusted":  (COLORS["green"], "-", 2),
    }
    for name, rets in portfolios.items():
        c, ls, lw = style[name]
        ax.plot(returns_matrix.index, np.cumprod(1 + rets),
                label=name, color=c, linestyle=ls, linewidth=lw)
    ax.set_title("Equity Curves"); ax.set_ylabel("Portfolio Value")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (0,1) Allocation comparison
    ax = axes[0, 1]
    x  = np.arange(len(pairs)); w = 0.25
    ax.bar(x - w, w_equal * 100, w, label="1/N",       color="#95A5A6", alpha=0.7)
    ax.bar(x,     w_markowitz * 100, w, label="Markowitz", color=COLORS["blue"], alpha=0.8)
    ax.bar(x + w, w_hurst * 100, w, label="Hurst-Adj", color=COLORS["green"], alpha=0.8)
    ax.set_title("Allocation Comparison"); ax.set_ylabel("Weight (%)")
    ax.set_xticks(x); ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=8)
    ax.axhline(100 / len(pairs), color="gray", ls=":", alpha=0.5)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # (1,0) Drawdown
    ax = axes[1, 0]
    for name, rets in [("Markowitz ML", portfolios["Markowitz ML"]),
                       ("Hurst-Adjusted", portfolios["Hurst-Adjusted"])]:
        cum = np.cumprod(1 + rets)
        dd  = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum) * 100
        col = COLORS["blue"] if "Markowitz" in name else COLORS["green"]
        ax.fill_between(returns_matrix.index, dd, 0, alpha=0.3, color=col, label=name)
        ax.plot(returns_matrix.index, dd, color=col, linewidth=0.8)
    ax.set_title("Drawdown (%)"); ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (1,1) Correlation heatmap
    ax = axes[1, 1]
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pairs))); ax.set_yticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(pairs, fontsize=8)
    for i in range(len(pairs)):
        for j in range(len(pairs)):
            v = corr_matrix.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.5 else "black")
    ax.set_title("Correlation Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.show()
    print(f"  [OK] {path}")


def plot_signal_analysis(data_dict: dict, pair: str = "EURUSD"):
    """Figure 2: 3-panel signal analysis — wavelet, HMM regimes, Hurst."""
    df = data_dict[pair]
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f"Signal Analysis — {pair}", fontsize=16, fontweight="bold", y=0.98)
    tail = 500

    # Wavelet denoising
    ax = axes[0]
    ax.plot(df.index[-tail:], df["Close"].iloc[-tail:],
            color="#BDC3C7", lw=0.8, label="Raw signal", alpha=0.8)
    ax.plot(df.index[-tail:], df["Close_Clean"].iloc[-tail:],
            color=COLORS["blue"], lw=1.5, label="Denoised (db4)")
    ax.set_title(f"Wavelet Denoising — {pair}"); ax.set_ylabel("Price")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # HMM regimes
    ax = axes[1]
    close = df["Close"].iloc[-tail:]
    regime_colors = {"BULL": COLORS["BULL"], "BEAR": COLORS["BEAR"], "NEUTRAL": COLORS["NEUTRAL"]}
    for lbl, col in regime_colors.items():
        mask = df["HMM_Regime_Label"].iloc[-tail:] == lbl
        ax.fill_between(df.index[-tail:], close.min(), close.max(),
                        where=mask, alpha=0.15, color=col, label=lbl)
    ax.plot(df.index[-tail:], close, color="black", lw=0.8)
    ax.set_title(f"HMM Regime Detection — {pair}"); ax.set_ylabel("Price")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(True, alpha=0.3)

    # Hurst exponent
    ax = axes[2]
    h = df["Hurst"].iloc[-tail:]
    ax.plot(df.index[-tail:], h, color=COLORS["purple"], lw=1.2)
    ax.axhline(0.5, color="red", ls="--", lw=1, alpha=0.7, label="H=0.5 (Random Walk)")
    ax.axhspan(0.45, 0.55, alpha=0.1, color="gray", label="Brownian zone")
    ax.fill_between(df.index[-tail:], h, 0.5,
                    where=h > 0.5, alpha=0.2, color=COLORS["green"], label="Trending")
    ax.fill_between(df.index[-tail:], h, 0.5,
                    where=h < 0.5, alpha=0.2, color=COLORS["red"], label="Mean-Reverting")
    ax.set_title(f"Rolling Hurst Exponent — {pair}"); ax.set_ylabel("H")
    ax.set_ylim(0.2, 0.9)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_signal_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.show()
    print(f"  [OK] {path}")


def plot_model_analysis(pairs: list, lstm_acc: dict, xgb_acc: dict,
                        sorted_features: list):
    """Figure 3: Feature importance + per-pair accuracy comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Analysis", fontsize=16, fontweight="bold", y=1.02)

    # Feature importance
    ax = axes[0]
    names = [f for f, _ in sorted_features]
    vals  = [v for _, v in sorted_features]
    cols  = [
        "#27AE60" if any(k in f for k in ("Clean", "Entropy", "Fractal", "Hurst"))
        else "#2C3E50"
        for f in names
    ]
    ax.barh(range(len(names)), vals, color=cols, alpha=0.85)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean Importance"); ax.set_title("XGBoost Feature Importance")
    ax.invert_yaxis()
    ax.legend(
        handles=[
            Patch(facecolor="#2C3E50", alpha=0.85, label="Classical"),
            Patch(facecolor="#27AE60", alpha=0.85, label="Novel features"),
        ], fontsize=8, loc="lower right",
    )
    ax.grid(True, alpha=0.3, axis="x")

    # Accuracy comparison
    ax = axes[1]
    x = np.arange(len(pairs)); w = 0.35
    ax.bar(x - w / 2, [lstm_acc[p] * 100 for p in pairs], w,
           label="Wavelet-LSTM", color=COLORS["blue"], alpha=0.8)
    ax.bar(x + w / 2, [xgb_acc[p] * 100 for p in pairs], w,
           label="XGBoost", color=COLORS["green"], alpha=0.8)
    ax.axhline(50, color="red", ls="--", lw=1, alpha=0.5, label="Random (50%)")
    ax.set_title("Accuracy by Pair and Model"); ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x); ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y"); ax.set_ylim(45, 60)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_model_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.show()
    print(f"  [OK] {path}")


def plot_allocation(pairs: list, w_markowitz: np.ndarray, w_hurst: np.ndarray,
                    Sigma: np.ndarray):
    """Figure 4: Pie charts (Markowitz vs Hurst-Adj) + risk contribution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Final Allocation & Risk Decomposition",
                 fontsize=14, fontweight="bold", y=1.02)
    pair_colors = [COLORS.get(p, "#888") for p in pairs]

    axes[0].pie(w_markowitz, labels=pairs, autopct="%1.1f%%", colors=pair_colors,
                startangle=140, pctdistance=0.8, textprops={"fontsize": 8})
    axes[0].set_title("Markowitz ML", fontsize=11, fontweight="bold")

    axes[1].pie(w_hurst, labels=pairs, autopct="%1.1f%%", colors=pair_colors,
                startangle=140, pctdistance=0.8, textprops={"fontsize": 8})
    axes[1].set_title("Hurst-Adjusted", fontsize=11, fontweight="bold")

    # Risk contribution
    ax = axes[2]
    port_std = np.sqrt(w_hurst @ Sigma @ w_hurst)
    mrc = (Sigma @ w_hurst) / port_std if port_std > 0 else np.zeros(len(pairs))
    rc  = w_hurst * mrc
    rc_pct = rc / np.sum(rc) * 100

    bars = ax.bar(pairs, rc_pct, color=pair_colors, alpha=0.85)
    ax.set_ylabel("Risk Contribution (%)"); ax.set_title("Risk Decomposition (Hurst-Adj)")
    ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=8)
    for bar, v in zip(bars, rc_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig4_allocation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.show()
    print(f"  [OK] {path}")


def plot_bl_performance(equity_curve, equity_dates, capital):
    """Figure 5: BL strategy equity curve + drawdown."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    df_eq = pd.DataFrame({"Equity": equity_curve}, index=equity_dates)

    ax1.plot(df_eq.index, df_eq["Equity"], color="#1f77b4", lw=1.5,
             label="BL Strategy (Real Rates)")
    ax1.axhline(TARGET_BUDGET, color="black", ls="--", alpha=0.5)
    ax1.set_title("Black-Litterman Strategy Performance",
                  fontsize=14, fontweight="bold")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    dd = (df_eq["Equity"] - df_eq["Equity"].cummax()) / df_eq["Equity"].cummax()
    ax2.plot(dd.index, dd, color="red", lw=1)
    ax2.fill_between(dd.index, dd, 0, color="red", alpha=0.2)
    ax2.set_ylabel("Drawdown"); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig5_bl_performance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.show()
    print(f"  [OK] {path}")


def plot_bl_scatter():
    """Figure 6: Risk/Return scatter (all pairs + strategy)."""
    returns = pd.read_csv(FILE_RETURNS, index_col=0, parse_dates=True)
    summary = pd.DataFrame()
    summary["Returns"]    = returns.mean() * 252
    summary["Volatility"] = returns.std() * np.sqrt(252)

    plt.figure(figsize=(12, 8))
    subset = summary.drop("STRATEGY", errors="ignore")
    sns.scatterplot(data=subset, x="Volatility", y="Returns",
                    s=100, color="gray", alpha=0.5, label="FX Pairs")

    if "STRATEGY" in summary.index:
        s = summary.loc["STRATEGY"]
        plt.scatter(s["Volatility"], s["Returns"], s=300,
                    color="green", edgecolors="black", label="STRATEGY")
        plt.text(s["Volatility"] + 0.003, s["Returns"],
                 "  BL STRATEGY", fontsize=11, fontweight="bold", color="green")

    for i in range(len(subset)):
        plt.text(subset.iloc[i]["Volatility"], subset.iloc[i]["Returns"],
                 subset.index[i], fontsize=8, alpha=0.7)

    plt.axhline(0, color="black")
    plt.title("Risk / Return Profile (Price + Carry)", fontsize=14)
    plt.xlabel("Volatility"); plt.ylabel("Total Return")
    plt.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, "fig6_bl_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.show()
    print(f"  [OK] {path}")


# ==============================================================================
#  SECTION 18 — SUMMARY
# ==============================================================================

def print_pipeline_summary():
    print_header("PIPELINE SUMMARY")
    print("""
  ┌──────────────────────────────────────────────────────────────────────┐
  │  FX Data  (7 core pairs + 17 Yahoo pairs + FRED rates)             │
  │       │                                                             │
  │       ├──────────────────┐                                          │
  │       ▼                  ▼                                          │
  │  Wavelet Denoising    Data Audit (NaN, date ranges, visual QA)     │
  │  (db4, VisuShrink)                                                 │
  │       │                                                             │
  │       ▼                                                             │
  │  Feature Engineering                                                │
  │  (13 features: classical + wavelet + complexity + Hurst)            │
  │       │                                                             │
  │       ├──────────────┬──────────────┐                               │
  │       ▼              ▼              ▼                               │
  │   Wavelet-LSTM    XGBoost     HMM (3 regimes)                      │
  │   (64→32, ES)     (100 trees)  (raw returns)                       │
  │       │              │              │                               │
  │       └──────┬───────┘              │                               │
  │              ▼                      │                               │
  │      Ensemble (50/50)    ×    HMM regime boost                     │
  │              │                                                      │
  │              ▼                                                      │
  │   μ = ensemble expected returns                                     │
  │              │                                                      │
  │              ├───────────────────────────────────┐                   │
  │              ▼                                   ▼                  │
  │   Markowitz: min w^T Σ w               Black-Litterman             │
  │   s.t. w^T μ ≥ r_target               (Momentum + Carry views)    │
  │              │                                   │                  │
  │              ▼                                   ▼                  │
  │   Hurst per-pair adjustment:           Weekly rebalanced backtest  │
  │   wᵢ *= 1 + 2·|Hᵢ − 0.5|             (no look-ahead, costs=5bps) │
  │              │                                   │                  │
  │              ▼                                   ▼                  │
  │   FINAL ALLOCATION               RISK AUDIT (Sharpe, VaR, DD)     │
  └──────────────────────────────────────────────────────────────────────┘
    """)
    print("  Implemented improvements:")
    print("   1. Wavelet denoising (db4, VisuShrink)        — Tang et al. 2023")
    print("   2. Complexity features (Shannon, Rényi)       — Garcin 2023")
    print("   3. LSTM architecture (64→32, EarlyStopping)   — Tang et al. 2023")
    print("   4. HMM on raw returns (avoids over-smoothing)")
    print("   5. Hurst per-pair weighting (|H−0.5| → boost)")
    print("   6. Black-Litterman with real carry data (FRED)")
    print("   7. Strict no-lookahead backtest (weekly rebal)")
    print("   8. Extended risk metrics (Sortino, Calmar, ES, Win Rate)")
    print("   9. Risk decomposition (marginal risk contribution)")
    print("  10. 1/N benchmark for fair comparison")
    print("=" * 80)


# ==============================================================================
#  MAIN ORCHESTRATOR
# ==============================================================================

def main():
    """
    Master entry point.  Two execution paths:

    PATH A — Real Data (Yahoo + FRED):
      download → audit → BL backtest → risk report → charts

    PATH B — Synthetic Data (always runs):
      generate → wavelet → features → Hurst → HMM → LSTM → XGBoost
      → ensemble → Markowitz → Hurst adjustment → metrics → charts
    """

    # ==================================================================
    #  PATH A : Real-data BL backtest (requires network + data files)
    # ==================================================================
    print_header("FX PORTFOLIO OPTIMIZATION ENGINE")
    print("  Wavelet-LSTM · XGBoost · HMM · Hurst · Black-Litterman")
    print("=" * 80)

    run_bl = False
    bl_equity, bl_dates, bl_rets, bl_capital = None, None, None, None

    try:
        # — Download (uncomment if first run) —
        # download_forex_prices()
        # download_fred_rates()

        prices, rates_daily = load_local_data()
        prices      = prices.loc["2015-01-01":]
        rates_daily = rates_daily.loc["2015-01-01":]

        run_data_audit(prices, rates_daily)

        returns_total, momentum_score, prices_bt = \
            compute_total_returns_and_momentum(prices, rates_daily)

        bl_equity, bl_dates, bl_rets, bl_capital = \
            run_backtest(returns_total, momentum_score)

        run_bl_risk_analysis(bl_capital)
        plot_bl_performance(bl_equity, bl_dates, bl_capital)
        plot_bl_scatter()
        run_bl = True

    except FileNotFoundError:
        print("\n  [INFO] Real data files not found — skipping BL backtest.")
        print("         Run download_forex_prices() + download_fred_rates() first.")
    except Exception as e:
        print(f"\n  [WARN] BL path failed: {e}")

    # ==================================================================
    #  PATH B : Synthetic-data ML pipeline (always runs)
    # ==================================================================
    data_dict = load_real_data_for_pipeline()

    # ENSUITE extraire les paires
    pairs = list(data_dict.keys())
    # Step 1 — Wavelet denoising
    apply_wavelet_denoising(data_dict, pairs)

    # Step 2 — Feature engineering
    apply_feature_engineering(data_dict, pairs)

    # Step 3 — Hurst exponent
    apply_hurst(data_dict, pairs)

    # Step 4 — HMM regime detection
    hmm_regimes, hmm_details = apply_hmm_regimes(data_dict, pairs)

    # Step 5 — Wavelet-LSTM
    lstm_preds, lstm_acc = train_lstm_models(data_dict, pairs)

    # Step 6 — XGBoost
    xgb_preds, xgb_acc, xgb_imp, sorted_fi = train_xgboost_models(data_dict, pairs)

    # Step 7 — Ensemble
    ensemble_preds = build_ensemble(pairs, lstm_preds, xgb_preds, hmm_regimes)

    # Step 8 — Markowitz optimisation
    w_markowitz, Sigma, mu, returns_matrix = run_markowitz(data_dict, pairs, ensemble_preds)

    # Step 9 — Hurst weight adjustment
    w_hurst = adjust_weights_hurst(data_dict, pairs, w_markowitz)

    # Step 10 — Performance metrics
    w_equal = np.ones(len(pairs)) / len(pairs)
    portfolios = {
        "1/N (benchmark)": returns_matrix.values @ w_equal,
        "Markowitz ML":    returns_matrix.values @ w_markowitz,
        "Hurst-Adjusted":  returns_matrix.values @ w_hurst,
    }
    corr = print_performance_metrics(portfolios, data_dict, pairs, returns_matrix)

    # Step 11 — Visualizations
    print_header("GENERATING CHARTS")
    plot_dashboard(portfolios, pairs, w_equal, w_markowitz, w_hurst,
                   returns_matrix, corr)
    plot_signal_analysis(data_dict, "EURUSD")
    plot_model_analysis(pairs, lstm_acc, xgb_acc, sorted_fi)
    plot_allocation(pairs, w_markowitz, w_hurst, Sigma)

    # Summary
    print_pipeline_summary()


# ==============================================================================
#  ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
