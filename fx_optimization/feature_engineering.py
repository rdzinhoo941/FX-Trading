"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  FEATURE ENGINEERING — 20 features par paire                                   ║
║                                                                                ║
║  Classiques (8) + Wavelet (3) + Entropie (2) + Fractal (3) + CS (1) + ATR(1) ║
║  + IC scoring (Information Coefficient) pour évaluer la qualité des features  ║
║                                                                                ║
║  Ref : Jansen ML4T ch.4 — Alpha Factor Research                               ║
║        Kakushadze (2016) — 101 Formulaic Alphas                               ║
║        Jansen ML4T ch.24 — Alpha Factor Library                               ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from config import FEATURE_COLS_BASE, FEATURE_COLS_CT, FEATURE_COLS_MT, FEATURE_COLS_LT, FEATURE_COLS
from utils_display import hdr, sub, tbl


# ════════════════════════════════════════════════════════════════════════════════
# INDICATEURS TECHNIQUES
# ════════════════════════════════════════════════════════════════════════════════

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range normalisé par le prix (ATR%).
    Mesure de volatilité locale plus précise que la vol réalisée simple.
    Ref : Wilder (1978), Jansen ML4T ch.24.
    """
    close = df['Close']
    # En l'absence de High/Low, on approxime TR = |Return_1d| × Close
    tr = close.pct_change().abs() * close
    return tr.rolling(period).mean() / close


def calculate_bollinger_pct(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Bollinger Band %B = (prix - lower) / (upper - lower).
    Valeur ∈ [0,1] : 0.5 = milieu bande, >1 = surchat, <0 = survente.
    Ref : Bollinger (1992), Jansen ML4T ch.24.
    """
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return (close - lower) / (upper - lower + 1e-10)


def shannon_entropy(series: np.ndarray, bins: int = 50) -> float:
    counts, _ = np.histogram(series[np.isfinite(series)], bins=bins)
    probs = counts / (counts.sum() + 1e-10)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def renyi_entropy(series: np.ndarray, alpha: float = 2.0, bins: int = 50) -> float:
    counts, _ = np.histogram(series[np.isfinite(series)], bins=bins)
    probs = counts / (counts.sum() + 1e-10)
    probs = probs[probs > 0]
    return float((1 / (1 - alpha)) * np.log2(np.sum(probs ** alpha)))


# ════════════════════════════════════════════════════════════════════════════════
# MOMENTUM CROSS-SECTIONNEL
# ════════════════════════════════════════════════════════════════════════════════

def compute_cross_sectional_momentum(data_dict: dict, lookback: int = 21) -> dict:
    """
    Momentum cross-sectionnel : rang z-score de chaque paire parmi toutes.
    Élimine l'effet EM (TRY, ZAR) qui dominent le momentum absolu.
    Ref : Jansen ML4T ch.4 — Cross-sectional factor construction.
    """
    # Construire une matrice de rendements lookback
    all_pairs = list(data_dict.keys())
    all_dates = data_dict[all_pairs[0]].index if all_pairs else pd.DatetimeIndex([])

    mom_matrix = pd.DataFrame(index=all_dates, columns=all_pairs, dtype=float)
    for pair, df in data_dict.items():
        r = df['Return_1d']
        mom_matrix[pair] = r.rolling(lookback).sum()

    # Z-score cross-sectionnel à chaque date
    cs_zscore = mom_matrix.sub(mom_matrix.mean(axis=1), axis=0).div(
        mom_matrix.std(axis=1) + 1e-10, axis=0
    )
    cs_zscore = cs_zscore.clip(-3, 3)

    for pair in all_pairs:
        if pair in data_dict:
            data_dict[pair]['Momentum_CS'] = cs_zscore[pair]

    return data_dict


# ════════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

def create_features(data_dict: dict) -> dict:
    hdr("FEATURE ENGINEERING (features CT/MT/LT par horizon)")

    for pair, df in data_dict.items():
        close = df['Close']
        clean = df.get('Close_Clean', close)
        clean_s = pd.Series(clean, index=df.index) if not isinstance(clean, pd.Series) else clean

        # ── Court terme (fenêtres 1-21j) ──
        df['Return_1d']        = close.pct_change()
        df['Return_5d']        = close.pct_change(5)
        df['RSI']              = calculate_rsi(close) / 100
        df['ATR_14']           = calculate_atr(df, period=14)
        df['Bollinger_Pct']    = calculate_bollinger_pct(close)
        df['RealizedVol_21d']  = df['Return_1d'].rolling(21).std() * np.sqrt(252)

        # ── Moyen terme (fenêtres 21-63j) ──
        df['Return_21d']       = close.pct_change(21)
        df['SMA_21']           = close / close.rolling(21).mean() - 1
        df['SMA_63']           = close / close.rolling(63).mean() - 1
        ema12, ema26           = close.ewm(span=12).mean(), close.ewm(span=26).mean()
        df['MACD']             = (ema12 - ema26) / close
        df['RealizedVol_63d']  = df['Return_1d'].rolling(63).std() * np.sqrt(252)

        # ── Long terme (fenêtres 63-252j) ──
        df['Return_63d']       = close.pct_change(63)
        df['SMA_126']          = close / close.rolling(126).mean() - 1
        df['RealizedVol_126d'] = df['Return_1d'].rolling(126).std() * np.sqrt(252)
        # Momentum 3 mois (63j) — signal MT/LT
        df['Momentum_3m']      = close.pct_change(63)
        # Momentum 12 mois (252j) — signal LT pur (Moskowitz, Ooi & Pedersen 2012)
        # On skip le dernier mois pour éviter le mean-reversion CT (convention standard)
        df['Momentum_12m']     = close.shift(21).pct_change(252 - 21)
        # Carry brut (placeholder — sera rempli par data_utils si disponible)
        if 'Carry_Raw' not in df.columns:
            df['Carry_Raw']    = 0.0

        # ── Wavelet (déjà calculé dans signal_processing) ──
        df['Return_Clean_1d'] = clean_s.pct_change()
        df['Return_Clean_5d'] = clean_s.pct_change(5)
        ema12c = clean_s.ewm(span=12).mean()
        ema26c = clean_s.ewm(span=26).mean()
        df['MACD_Clean']      = (ema12c - ema26c) / clean_s

        # ── Entropie roulante 100j ──
        df['Shannon_Entropy'] = df['Return_1d'].rolling(100).apply(
            lambda x: shannon_entropy(x), raw=True
        )
        df['Renyi_Entropy'] = df['Return_1d'].rolling(100).apply(
            lambda x: renyi_entropy(x), raw=True
        )

        # ── Momentum CS (placeholder — calculé après) ──
        if 'Momentum_CS' not in df.columns:
            df['Momentum_CS'] = 0.0

        # ── Targets par horizon ──
        df['Target_Direction']    = (df['Return_1d'].shift(-1) > 0).astype(int)   # CT J+1
        df['Target_Return']       = df['Return_1d'].shift(-1)                      # CT régression
        df['Target_Direction_21d']= (df['Return_1d'].rolling(21).sum().shift(-21) > 0).astype(int)  # MT
        df['Target_Return_21d']   = df['Return_1d'].rolling(21).sum().shift(-21)   # MT régression
        df['Target_Direction_63d']= (df['Return_1d'].rolling(63).sum().shift(-63) > 0).astype(int)  # LT
        df['Target_Return_63d']   = df['Return_1d'].rolling(63).sum().shift(-63)   # LT régression

    # Momentum cross-sectionnel (nécessite toutes les paires)
    data_dict = compute_cross_sectional_momentum(data_dict)

    print(f"  CT: {len(FEATURE_COLS_CT)} features, target J+1")
    print(f"  MT: {len(FEATURE_COLS_MT)} features, target J+21")
    print(f"  LT: {len(FEATURE_COLS_LT)} features, target J+63")
    return data_dict


# ════════════════════════════════════════════════════════════════════════════════
# IC SCORING — Information Coefficient (Alphalens-style)
# ════════════════════════════════════════════════════════════════════════════════
# Ref : Jansen ML4T ch.4 — Evaluating alpha factors with Alphalens
#       Grinold & Kahn (1999) — Active Portfolio Management

def compute_ic(data_dict: dict, feature_cols: list = None,
               target_col: str = 'Target_Return',
               lookback: int = 252) -> pd.DataFrame:
    """
    Calcule l'Information Coefficient (rank correlation de Spearman)
    entre chaque feature et le rendement futur.

    IC > 0.05 = signal exploitable (règle empirique Grinold & Kahn).
    IC > 0.10 = signal fort.

    Retourne un DataFrame avec [feature, IC_mean, IC_std, IC_IR, pct_positive].
    """
    sub("IC SCORING (Spearman rank correlation)")

    if feature_cols is None:
        feature_cols = FEATURE_COLS_MT

    ic_results = {f: [] for f in feature_cols}

    for pair, df in data_dict.items():
        valid = df[feature_cols + [target_col]].dropna()
        if len(valid) < lookback:
            continue
        # IC roulant sur les 252 derniers jours
        window = valid.tail(lookback)
        for feat in feature_cols:
            try:
                ic, _ = spearmanr(window[feat], window[target_col])
                if np.isfinite(ic):
                    ic_results[feat].append(ic)
            except Exception:
                pass

    rows = []
    summary = []
    for feat in feature_cols:
        ics = ic_results[feat]
        if not ics:
            continue
        ic_arr = np.array(ics)
        ic_mean   = np.mean(ic_arr)
        ic_std    = np.std(ic_arr) + 1e-10
        ic_ir     = ic_mean / ic_std          # Information Ratio du signal
        pct_pos   = np.mean(ic_arr > 0)
        quality   = "★★★" if abs(ic_mean) > 0.10 else ("★★" if abs(ic_mean) > 0.05 else "★")
        rows.append([feat, f"{ic_mean:+.4f}", f"{ic_std:.4f}",
                     f"{ic_ir:+.3f}", f"{pct_pos:.0%}", quality])
        summary.append({
            'feature': feat, 'IC_mean': ic_mean, 'IC_std': ic_std,
            'IC_IR': ic_ir, 'pct_positive': pct_pos
        })

    if rows:
        tbl(["Feature", "IC mean", "IC std", "IC/IR", "% positif", "Qualité"], rows)

    return pd.DataFrame(summary).sort_values('IC_IR', ascending=False)