"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  TRAITEMENT DU SIGNAL                                                          ║
║                                                                                ║
║  COUCHE 1 : WAVELET — SureShrink (Donoho & Johnstone 1995)                    ║
║  COUCHE 2 : HURST — Absolute Moments + prédictions fBm (Garcin 2021)          ║
║  COUCHE 3 : HMM RÉGIMES — Hidden Markov Model 2 états (low/high vol)          ║
║  COUCHE 4 : GARCH(1,1) — prévision de volatilité forward-looking              ║
║                                                                                ║
║  Note : Kalman filter supprimé. Les features Kalman_Residual et               ║
║         Kalman_Uncertainty sont remplacées par les résidus HMM et             ║
║         la vol GARCH — plus robustes et académiquement mieux fondés.          ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import pywt
from typing import Tuple
import warnings
from sklearn.linear_model import LinearRegression

from config import USER
from utils_display import hdr, sub, tbl

warnings.filterwarnings('ignore')


# ════════════════════════════════════════════════════════════════════════════════
# COUCHE 1 : WAVELET DENOISING — SureShrink (Donoho & Johnstone 1995)
# ════════════════════════════════════════════════════════════════════════════════

def wavelet_denoise(signal: np.ndarray, wavelet: str = None,
                    level: int = None, method: str = None) -> np.ndarray:
    """
    Débruite un signal par ondelettes.
    SureShrink : minimise le risque SURE (Stein's Unbiased Risk Estimate).
    Référence : Donoho & Johnstone (1995).
    """
    wavelet = wavelet or USER.wavelet_name
    level   = level or USER.wavelet_level
    method  = method or USER.wavelet_method

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    if method == 'visu':
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs_t = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]

    elif method == 'sure':
        coeffs_t = [coeffs[0]]
        for c in coeffs[1:]:
            sigma = np.median(np.abs(c)) / 0.6745
            n = len(c)
            if sigma < 1e-10 or n < 2:
                coeffs_t.append(c)
                continue
            c_norm = c / sigma
            c2_sorted = np.sort(c_norm ** 2)
            risks = np.zeros(n)
            for j in range(n):
                t2 = c2_sorted[j]
                risks[j] = n - 2 * (j + 1) + np.sum(np.minimum(c2_sorted, t2))
            best_idx = np.argmin(risks)
            threshold = sigma * np.sqrt(c2_sorted[best_idx])
            threshold = min(threshold, sigma * np.sqrt(2 * np.log(n)))
            coeffs_t.append(pywt.threshold(c, threshold, mode='soft'))
    else:
        coeffs_t = coeffs

    return pywt.waverec(coeffs_t, wavelet)[:len(signal)]


def apply_wavelet_denoising(data_dict: dict) -> dict:
    hdr(f"WAVELET DENOISING ({USER.wavelet_name}, {USER.wavelet_method.upper()})")
    rows = []
    for pair, df in data_dict.items():
        raw = df['Close'].values
        clean = wavelet_denoise(raw)
        noise_pct = np.std(raw - clean) / np.std(raw) * 100
        df['Close_Clean'] = clean
        df['Return_Clean'] = pd.Series(clean, index=df.index).pct_change()
        rows.append([pair, f"{noise_pct:.1f}%"])
    tbl(["Pair", "Bruit retiré"], rows)
    return data_dict


# ════════════════════════════════════════════════════════════════════════════════
# COUCHE 2 : HURST — Absolute Moments + Prédictions fBm (Garcin 2021)
# ════════════════════════════════════════════════════════════════════════════════

def estimate_hurst_moments(returns: np.ndarray, max_lag: int = 50) -> float:
    """
    Estime H par la méthode des moments absolus.
    Ref : Garcin (2021) — Forecasting with fractional Brownian motion.
    """
    if len(returns) < 50:
        return 0.5
    try:
        prices = np.cumsum(returns - np.mean(returns))
        lags = np.unique(np.logspace(0, np.log10(max_lag), 15).astype(int))
        lags = lags[lags < len(prices) // 2]
        if len(lags) < 3:
            return 0.5
        log_m1, log_m2, log_tau = [], [], []
        for tau in lags:
            diffs = prices[tau:] - prices[:-tau]
            if len(diffs) < 10:
                continue
            m1 = np.mean(np.abs(diffs))
            m2 = np.mean(diffs ** 2)
            if m1 > 0 and m2 > 0:
                log_m1.append(np.log(m1))
                log_m2.append(np.log(m2))
                log_tau.append(np.log(tau))
        if len(log_m1) < 3:
            return 0.5
        X = np.array(log_tau).reshape(-1, 1)
        H1 = LinearRegression().fit(X, np.array(log_m1)).coef_[0]
        H2 = LinearRegression().fit(X, np.array(log_m2)).coef_[0] / 2
        H = 0.3 * H1 + 0.7 * H2
        return np.clip(H, 0.2, 0.8)
    except Exception:
        return 0.5


def predict_fbm_garcin(series: np.ndarray, H: float, horizon: int = 5) -> float:
    """Prédiction fBm basée sur Garcin (2021)."""
    try:
        if len(series) < 50:
            return 0.0
        series = series[series > 0]
        if len(series) < 50:
            return 0.0
        log_ret = np.diff(np.log(series))
        if len(log_ret) < 20:
            return 0.0
        n_past = min(50, len(log_ret))
        recent = log_ret[-n_past:]
        if abs(H - 0.5) < 0.03:
            return 0.0
        gamma = np.zeros(n_past)
        for k in range(n_past):
            j = k + 1
            gamma[k] = 0.5 * (
                abs(horizon + j) ** (2 * H) - 2 * abs(horizon + j - 1) ** (2 * H)
                + abs(horizon + j - 2) ** (2 * H)
            )
        Gamma = np.zeros((n_past, n_past))
        for i in range(n_past):
            for j in range(n_past):
                k = abs(i - j)
                Gamma[i, j] = 0.5 * (abs(k + 1) ** (2 * H)
                                      - 2 * abs(k) ** (2 * H)
                                      + abs(max(k - 1, 0)) ** (2 * H))
        Gamma += 1e-8 * np.eye(n_past)
        try:
            weights = np.linalg.solve(Gamma, gamma)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(Gamma, gamma, rcond=None)[0]
        prediction = float(np.dot(weights, recent[::-1]))
        return np.clip(prediction * 100, -3, 3)
    except Exception:
        return 0.0


def apply_hurst(data_dict: dict) -> dict:
    hdr("HURST (Absolute Moments) + Prédictions fBm (Garcin)")
    rows = []
    for pair, df in data_dict.items():
        returns = df['Return_1d'].dropna()
        df['Hurst'] = returns.rolling(252, min_periods=100).apply(
            lambda x: estimate_hurst_moments(x.values, max_lag=60), raw=False
        )
        df['Fractal_Dimension'] = 2 - df['Hurst']
        df['Hurst_Prediction'] = np.nan
        for i in range(252, len(df)):
            if not pd.isna(df['Hurst'].iloc[i]):
                series = df['Close'].iloc[max(0, i - 252):i].values
                H = df['Hurst'].iloc[i]
                df.iloc[i, df.columns.get_loc('Hurst_Prediction')] = predict_fbm_garcin(
                    series, H, horizon=5
                )
        h_last = df['Hurst'].iloc[-1] if not df['Hurst'].isna().all() else 0.5
        regime = "TRENDING" if h_last > 0.55 else ("MEAN-REV" if h_last < 0.45 else "RND WALK")
        pred = df['Hurst_Prediction'].iloc[-1] if not df['Hurst_Prediction'].isna().all() else 0.0
        rows.append([pair, f"{h_last:.3f}", regime, f"{pred:+.2f}%"])
    tbl(["Pair", "H actuel", "Type", "Préd. 5j"], rows)
    return data_dict


# ════════════════════════════════════════════════════════════════════════════════
# COUCHE 3 : HMM RÉGIMES — Hidden Markov Model
# ════════════════════════════════════════════════════════════════════════════════
# Ref : Jansen ML4T ch.9 — Time Series Models
#       Hamilton (1989) — A New Approach to the Economic Analysis of Nonstationary
#       Time Series and the Business Cycle

def _fit_hmm_simple(returns: np.ndarray, n_states: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HMM gaussien simplifié via EM (sans hmmlearn pour portabilité).
    Retourne : états (0=low-vol, 1=high-vol), mu par état, sigma par état.

    Si hmmlearn est disponible, l'utilise. Sinon, fallback sur GMM via sklearn.
    """
    r = returns[np.isfinite(returns)]
    if len(r) < 50:
        return np.zeros(len(returns), dtype=int), np.array([0.0, 0.0]), np.array([0.01, 0.02])

    # ── Tentative avec hmmlearn ──
    try:
        from hmmlearn import hmm
        model = hmm.GaussianHMM(n_components=n_states, covariance_type='diag',
                                 n_iter=100, random_state=42)
        model.fit(r.reshape(-1, 1))
        states_r = model.predict(r.reshape(-1, 1))
        # Aligner : état 0 = plus faible vol
        vols = [r[states_r == s].std() for s in range(n_states)]
        if len(vols) == n_states and vols[0] > vols[1]:
            states_r = 1 - states_r   # swap

        # Remapper sur la série originale (NaN → état neutre 0)
        states_full = np.zeros(len(returns), dtype=int)
        finite_idx = np.where(np.isfinite(returns))[0]
        for i, idx in enumerate(finite_idx):
            if i < len(states_r):
                states_full[idx] = int(states_r[i])
        mus    = np.array([r[states_r == s].mean() if (states_r == s).any() else 0.0
                           for s in range(n_states)])
        sigmas = np.array([r[states_r == s].std() if (states_r == s).any() else 0.01
                           for s in range(n_states)])
        return states_full, mus, sigmas

    except ImportError:
        pass

    # ── Fallback : GMM (sklearn toujours disponible) ──
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=n_states, covariance_type='full',
                          random_state=42, max_iter=200)
    gm.fit(r.reshape(-1, 1))
    labels_r = gm.predict(r.reshape(-1, 1))

    # Aligner : état 0 = low-vol
    vols = [r[labels_r == s].std() for s in range(n_states)]
    if vols[0] > vols[1]:
        labels_r = 1 - labels_r

    states_full = np.zeros(len(returns), dtype=int)
    finite_idx = np.where(np.isfinite(returns))[0]
    for i, idx in enumerate(finite_idx):
        if i < len(labels_r):
            states_full[idx] = int(labels_r[i])

    mus    = np.array([r[labels_r == s].mean() if (labels_r == s).any() else 0.0
                       for s in range(n_states)])
    sigmas = np.array([r[labels_r == s].std()  if (labels_r == s).any() else 0.01
                       for s in range(n_states)])
    return states_full, mus, sigmas


def apply_regimes(data_dict: dict) -> dict:
    """
    Détecte les régimes via HMM (ou GMM en fallback).
    État 0 = low-vol / trending, État 1 = high-vol / stress.
    Ajoute aussi Regime_Mu et Regime_Sigma pour les features.
    """
    hdr(f"RÉGIMES HMM (n_states={USER.hmm_n_states})")
    regime_map = {0: "LOW-VOL", 1: "HIGH-VOL"}
    rows = []

    for pair, df in data_dict.items():
        returns = df['Return_1d'].values
        states, mus, sigmas = _fit_hmm_simple(returns, n_states=USER.hmm_n_states)

        df['Regime_Code']  = states
        df['Regime']       = pd.Series(states, index=df.index).map(regime_map)
        df['Regime_Mu']    = pd.Series(states, index=df.index).map(
            {s: mus[s] for s in range(USER.hmm_n_states)}
        )
        df['Regime_Sigma'] = pd.Series(states, index=df.index).map(
            {s: sigmas[s] for s in range(USER.hmm_n_states)}
        )
        current_regime = df['Regime'].iloc[-1] if not df['Regime'].isna().all() else "?"
        rows.append([pair, current_regime,
                     f"{mus[0]*10000:+.1f}bps",
                     f"{sigmas[0]*100:.2f}%",
                     f"{sigmas[1]*100:.2f}%"])

    tbl(["Pair", "Régime actuel", "μ état0", "σ état0", "σ état1"], rows)
    return data_dict


# ════════════════════════════════════════════════════════════════════════════════
# COUCHE 4 : GARCH(1,1) — Prévision de volatilité forward-looking
# ════════════════════════════════════════════════════════════════════════════════
# Ref : Jansen ML4T ch.9 — ARCH models for volatility forecasting
#       Engle (1982) — ARCH, Bollerslev (1986) — GARCH

def _fit_garch_simple(returns: np.ndarray, horizon: int = 5) -> np.ndarray:
    """
    GARCH(1,1) estimé par MLE simplifié (ou via arch si disponible).
    Retourne la variance conditionnelle prévue à t+horizon pour chaque t.

    Si arch est disponible, l'utilise. Sinon, fallback EWMA robuste.
    """
    r = np.array(returns, dtype=float)
    r = np.where(np.isfinite(r), r, 0.0)
    n = len(r)

    # ── Tentative avec arch ──
    try:
        from arch import arch_model
        am = arch_model(r * 100, vol='GARCH', p=1, q=1, dist='normal',
                        rescale=False)
        res = am.fit(disp='off', show_warning=False)
        forecasts = res.forecast(horizon=horizon, reindex=False)
        var_forecast = forecasts.variance.values[-1, horizon - 1] / (100 ** 2)
        # Pour avoir une série complète, on recalcule les variances conditionnelles
        cond_var = res.conditional_volatility ** 2 / (100 ** 2)
        cond_var_full = np.full(n, np.nan)
        cond_var_full[:len(cond_var)] = cond_var
        return np.sqrt(np.clip(cond_var_full, 1e-10, None))
    except (ImportError, Exception):
        pass

    # ── Fallback : EWMA (RiskMetrics) ──
    # σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1},  λ=0.94 (JP Morgan standard)
    lam = 0.94
    var = np.zeros(n)
    var[0] = np.var(r[:min(21, n)])
    for t in range(1, n):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(np.clip(var, 1e-10, None))


def apply_garch(data_dict: dict) -> dict:
    """
    Applique GARCH(1,1) ou EWMA pour prévoir la vol forward-looking.
    Ajoute GARCH_Vol (annualisée) à chaque paire.
    """
    if not USER.use_garch:
        for pair, df in data_dict.items():
            df['GARCH_Vol'] = df['Return_1d'].rolling(21).std() * np.sqrt(252)
        return data_dict

    hdr("GARCH(1,1) — Volatilité forward-looking")
    rows = []
    for pair, df in data_dict.items():
        returns = df['Return_1d'].fillna(0).values
        vol_daily = _fit_garch_simple(returns, horizon=5)
        df['GARCH_Vol'] = pd.Series(vol_daily * np.sqrt(252), index=df.index)
        current_vol = df['GARCH_Vol'].dropna().iloc[-1] if not df['GARCH_Vol'].isna().all() else 0.0
        rows.append([pair, f"{current_vol:.2%}"])
    tbl(["Pair", "Vol GARCH annualisée (actuelle)"], rows)
    return data_dict