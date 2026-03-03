"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  STRATÉGIES PAR HORIZON                                                        ║
║                                                                                ║
║  CT  (1-5j)   : Ridge / LightGBM / LightGBM+Hurst                            ║
║  MT  (5-21j)  : ML Ensemble — LSTM + Transformer + GB                         ║
║  LT  (21-63j) : Carry+Mom BL / Random Forest BL / MLP macro BL               ║
║  EP  standalone : Entropy Pooling sur vues brutes ou ML                       ║
║  COMBINÉ      : Entropy Pooling fusion CT+MT+LT + Vol Targeting               ║
║                                                                                ║
║  Références :                                                                  ║
║    Meucci (2008) — Entropy Pooling                                            ║
║    Black & Litterman (1992)                                                   ║
║    Jansen ML4T ch.5 — Portfolio Optimization                                 ║
║    Rockafellar & Uryasev (2000) — CVaR optimization                          ║
║    DMVFEP (2024) — Deep Multi-View Factor Entropy Pooling                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Tuple

from config import (
    BACKTEST_START, TRADING_DAYS, PAIR_MAP, TICKER_TO_PAIR, USER,
    FEATURE_COLS_LT
)
from metrics_performance import StrategyResult
from utils_display import hdr, sub, tbl


# ════════════════════════════════════════════════════════════════════════════════
# UTILITAIRES COMMUNS
# ════════════════════════════════════════════════════════════════════════════════

def _get_signal_at(res_dict: dict, dt, default: float = 0.5) -> float:
    """Récupère la dernière proba disponible avant dt (no look-ahead)."""
    series = res_dict.get('prediction_series', pd.Series(dtype=float))
    if isinstance(series, pd.Series) and len(series) > 0:
        past = series.loc[:dt]
        if len(past) > 0:
            return float(past.iloc[-1])
    return default


def _vol_target_scale(w: np.ndarray, cov: np.ndarray,
                      target_vol: float, max_lev: float) -> np.ndarray:
    """Applique le vol targeting : scale w pour atteindre target_vol."""
    vol_daily = np.sqrt(max(float(w.T @ cov @ w), 0))
    vol_ann   = vol_daily * np.sqrt(TRADING_DAYS)
    if vol_ann > 1e-10:
        lev = min(target_vol / vol_ann, max_lev)
        w   = w * lev
    return w


# ════════════════════════════════════════════════════════════════════════════════
# ENTROPY POOLING (Meucci 2008)
# ════════════════════════════════════════════════════════════════════════════════

def entropy_pooling(R: np.ndarray, mu_target: np.ndarray,
                    p: Optional[np.ndarray] = None,
                    tol: float = 1e-10, maxiter: int = 200) -> Tuple[np.ndarray, bool]:
    """
    Entropy Pooling (Meucci, 2008).
    Trouve la distribution q la plus proche de p (au sens KL)
    telle que E_q[R] = mu_target.
    """
    S, I = R.shape
    if p is None:
        p = np.full(S, 1.0 / S)
    p = p / p.sum()
    b = np.zeros(I)

    def compute_q(bvec):
        z = -R @ bvec
        z = z - np.max(z)
        w = p * np.exp(z)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            return None
        return w / s

    q = compute_q(b)
    if q is None:
        return p.copy(), False

    best_err, best_q = float('inf'), q.copy()
    ridge = 1e-8

    for it in range(maxiter):
        mu_q  = R.T @ q
        diff  = mu_q - mu_target
        err   = float(np.linalg.norm(diff))
        if err < tol:
            return q, True
        if err < best_err:
            best_err, best_q = err, q.copy()
        X   = R - mu_q
        Cov = (X.T * q) @ X + ridge * np.eye(I)
        try:
            step = np.linalg.solve(Cov, diff)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(Cov, diff, rcond=None)[0]
        alpha = 1.0
        for _ in range(20):
            q_new = compute_q(b + alpha * step)
            if q_new is not None:
                err_new = float(np.linalg.norm(R.T @ q_new - mu_target))
                if err_new < err:
                    b += alpha * step
                    q = q_new
                    break
            alpha *= 0.5

    return best_q, best_err < 1e-6


def _ep_posterior_to_weights(R: np.ndarray, mu_view: np.ndarray,
                              risk_aversion: float, max_weight: float,
                              max_leverage: float) -> np.ndarray:
    """EP → posterior → portefeuille tangent."""
    q, converged = entropy_pooling(R, mu_view)
    mu_post = R.T @ q
    X       = R - mu_post
    Sigma   = (X.T * q) @ X + 1e-6 * np.eye(R.shape[1])
    w = np.linalg.pinv(Sigma) @ mu_post
    w = np.clip(w, -max_weight, max_weight)
    gross = np.sum(np.abs(w))
    if gross > max_leverage:
        w *= max_leverage / gross
    return w


# ════════════════════════════════════════════════════════════════════════════════
# CVaR OPTIMIZATION (Rockafellar & Uryasev 2000)
# ════════════════════════════════════════════════════════════════════════════════
# Minimise les pertes extrêmes (queue gauche) plutôt que la variance.
# Plus robuste que MV pour les distributions non-gaussiennes (FX, EM).

def cvar_optimize(R: np.ndarray, mu: np.ndarray,
                  alpha: float = 0.05,
                  max_weight: float = 0.25,
                  max_leverage: float = 1.5,
                  risk_aversion: float = 2.5) -> np.ndarray:
    """
    Optimisation CVaR linéaire (Rockafellar & Uryasev 2000).
    Minimise : -mu'w + λ × CVaR_α(w)
    """
    S, n = R.shape
    k    = int(np.floor(alpha * S))
    k    = max(k, 1)

    def obj(w):
        port_ret = R @ w
        sorted_ret = np.sort(port_ret)
        cvar = -np.mean(sorted_ret[:k])
        return -float(mu @ w) + risk_aversion * cvar

    bounds = [(-max_weight, max_weight)] * n
    cons   = [{'type': 'ineq', 'fun': lambda w: max_leverage - np.sum(np.abs(w))}]
    w0     = np.zeros(n)
    result = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons,
                      options={'maxiter': 200, 'ftol': 1e-8})
    return result.x if result.success else w0


# ════════════════════════════════════════════════════════════════════════════════
# BLACK-LITTERMAN POSTERIOR
# ════════════════════════════════════════════════════════════════════════════════

def compute_bl_posterior(mu_eq: np.ndarray, cov: np.ndarray,
                          P: np.ndarray, Q: np.ndarray,
                          tau: float = 0.05) -> np.ndarray:
    """
    Black-Litterman posterior.
    P : matrice des vues (n_vues × n_actifs)
    Q : vecteur des rendements espérés des vues
    tau : confiance dans le prior (0.05 standard)
    """
    n = len(mu_eq)
    tau_cov = tau * cov
    # Omega : incertitude sur les vues — proportionnelle à tau × P × Σ × P'
    omega   = np.diag(np.diag(P @ tau_cov @ P.T)) + 1e-10 * np.eye(len(Q))
    try:
        tau_inv   = np.linalg.inv(tau_cov + 1e-10 * np.eye(n))
        omega_inv = np.linalg.inv(omega)
        M         = np.linalg.inv(tau_inv + P.T @ omega_inv @ P)
        return M @ (tau_inv @ mu_eq + P.T @ omega_inv @ Q)
    except np.linalg.LinAlgError:
        return mu_eq


def _build_bl_views_from_ml(assets: list, ml_res: dict, dt,
                              n_views: int = 5,
                              view_scale: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit P, Q à partir des signaux ML pour BL.
    Les vues sont pondérées par la confiance (accuracy OOS) de chaque modèle.
    Ref : DMVFEP (2024) — ML-generated views for BL.
    """
    n = len(assets)
    signals = np.zeros(n)

    for i, a in enumerate(assets):
        pair = TICKER_TO_PAIR.get(a, a.replace("=X", ""))
        prob = _get_signal_at(ml_res.get(pair, {}), dt)
        signals[i] = (prob - 0.5) * 2.0   # ∈ [-1, 1]

    ranked    = np.argsort(signals)
    n_views   = min(n_views, max(2, n // 4))
    P_rows, Q_rows = [], []

    for idx in ranked[-n_views:]:     # Vues longues
        row       = np.zeros(n)
        row[idx]  = 1.0
        P_rows.append(row)
        Q_rows.append(np.tanh(signals[idx]) * view_scale)

    for idx in ranked[:n_views]:      # Vues courtes
        row       = np.zeros(n)
        row[idx]  = -1.0
        P_rows.append(row)
        Q_rows.append(np.tanh(signals[idx]) * view_scale)

    return np.array(P_rows), np.array(Q_rows)


# ════════════════════════════════════════════════════════════════════════════════
# BENCHMARK : 1/N
# ════════════════════════════════════════════════════════════════════════════════

def strategy_equal_weight(ret: pd.DataFrame) -> StrategyResult:
    return StrategyResult(
        name="0. Equal Weight (1/N)",
        daily_returns=ret.mean(axis=1),
        description="Poids égaux, benchmark naïf (DeMiguel 2009)."
    ).compute_metrics()


# ════════════════════════════════════════════════════════════════════════════════
# COURT TERME — ML driven (Ridge / LightGBM / LightGBM+Hurst)
# ════════════════════════════════════════════════════════════════════════════════

def strategy_short_term(ret: pd.DataFrame, data_dict: dict,
                         ct_res: dict = None) -> StrategyResult:
    """
    Court terme.

    Si ct_res est fourni (Ridge ou LightGBM) : utilise le signal ML pour le sizing.
    Sinon : fallback sur le z-score mean-reversion original.

    Rebalancement tous les wf_retrain_freq / ~2 jours.
    Ref : Jansen ML4T ch.7 (Ridge), ch.12 (LightGBM).
    """
    pipeline = USER.get_active_pipelines()['CT']
    has_ml   = ct_res is not None and len(ct_res) > 0

    label = f"1. Court Terme ({pipeline['model']})" if has_ml else "1. Court Terme (MeanRev)"
    hdr(f"COURT TERME — {label}")

    assets    = ret.columns.tolist()
    n         = len(assets)
    dates     = ret.loc[BACKTEST_START:].index
    rebal_f   = pipeline['rebal_freq']
    lev_scale = pipeline['leverage']

    port_returns = pd.Series(0.0, index=dates, dtype=float)
    w_prev       = np.zeros(n)

    for t_idx, dt in enumerate(dates):
        loc = ret.index.get_loc(dt)
        port_returns.loc[dt] = float(w_prev @ ret.iloc[loc].values)

        if t_idx % rebal_f != 0:
            continue

        w_new = np.zeros(n)

        for i, a in enumerate(assets):
            pair = TICKER_TO_PAIR.get(a, a.replace("=X", ""))
            if pair not in data_dict:
                continue
            df    = data_dict[pair]
            avail = df.loc[:dt]
            if len(avail) < 63:
                continue

            if has_ml and pair in ct_res:
                # ── Signal ML (Ridge ou LightGBM) ──
                prob   = _get_signal_at(ct_res[pair], dt)
                signal = (prob - 0.5) * 2.0   # ∈ [-1, 1]

                # Filtre Hurst pour lgbm_hurst_ct : atténuer en régime trending
                if pipeline['model'] == 'lgbm_hurst_ct' and 'Hurst' in df.columns:
                    h = avail['Hurst'].dropna()
                    if len(h) > 0:
                        H = float(h.iloc[-1])
                        if H > 0.55:
                            signal *= 0.3   # mean-rev peu probable en trending

                w_new[i] = np.tanh(signal * 1.5)

            else:
                # ── Fallback : z-score mean-reversion original ──
                ret_1d = avail['Return_1d'].dropna()
                if len(ret_1d) < 21:
                    continue
                mu_21, std_21 = ret_1d.iloc[-21:].mean(), ret_1d.iloc[-21:].std()
                if std_21 < 1e-10:
                    continue
                z_score      = mu_21 / std_21
                signal_zscore = -np.tanh(z_score * 1.5)

                H, hurst_pred = 0.5, 0.0
                if 'Hurst' in df.columns:
                    h = avail['Hurst'].dropna()
                    if len(h) > 0:
                        H = float(h.iloc[-1])
                if 'Hurst_Prediction' in df.columns:
                    hp = avail['Hurst_Prediction'].dropna()
                    if len(hp) > 0:
                        hurst_pred = float(hp.iloc[-1])
                hurst_factor  = np.clip(1.0 - 2.0 * (H - 0.5), 0.3, 1.5)
                signal_garcin = np.tanh(hurst_pred / 3.0) if abs(hurst_pred) > 0.1 else 0.0
                raw_signal    = 0.75 * signal_zscore + 0.25 * signal_garcin
                w_new[i]      = raw_signal * hurst_factor

        # Normaliser sous contrainte levier CT
        max_lev = USER.max_leverage * lev_scale
        gross   = np.sum(np.abs(w_new))
        if gross > max_lev:
            w_new *= max_lev / gross

        # Coûts de transaction
        turnover = np.sum(np.abs(w_new - w_prev))
        if turnover > 0.01:
            port_returns.loc[dt] -= turnover * USER.transaction_cost
        w_prev = w_new

    return StrategyResult(
        name=label,
        daily_returns=port_returns.dropna(),
        description=f"CT {pipeline['model']}, rebal/{rebal_f}j, lev={lev_scale:.0%}×max"
    ).compute_metrics()


# ════════════════════════════════════════════════════════════════════════════════
# MOYEN TERME — ML Ensemble (GB + LSTM + Transformer)
# ════════════════════════════════════════════════════════════════════════════════

def strategy_medium_term(ret: pd.DataFrame, data_dict: dict,
                          gb_res: dict, lstm_res: dict,
                          trans_res: dict) -> StrategyResult:
    """
    Moyen terme : ensemble GB + Transformer + LSTM.
    Pipeline agressif : ajoute EP interne calibré par accuracy + BL/MV.
    Ref : Jansen ML4T ch.19, DMVFEP (2024).
    """
    pipeline  = USER.get_active_pipelines()['MT']
    use_ep    = pipeline['ep']
    use_blmv  = pipeline['bl_mv']
    lev_scale = pipeline['leverage']
    rebal_freq = pipeline['rebal_freq']   # 10j bi-mensuel ou 21j mensuel

    hdr(f"MOYEN TERME — {pipeline['model']} | rebal/{rebal_freq}j | EP={use_ep} BL/MV={use_blmv}")

    assets = ret.columns.tolist()
    n      = len(assets)
    dates  = ret.loc[BACKTEST_START:].index

    port_returns = pd.Series(0.0, index=dates, dtype=float)
    w_prev       = np.zeros(n)

    for t_idx, dt in enumerate(dates):
        loc = ret.index.get_loc(dt)
        port_returns.loc[dt] = float(w_prev @ ret.iloc[loc].values)

        if t_idx % rebal_freq != 0:
            continue

        w_new   = np.zeros(n)
        signals = np.zeros(n)

        for i, a in enumerate(assets):
            pair = TICKER_TO_PAIR.get(a, a.replace("=X", ""))
            if pair not in data_dict:
                continue

            gb_prob    = _get_signal_at(gb_res.get(pair, {}),    dt)
            lstm_prob  = _get_signal_at(lstm_res.get(pair, {}),  dt)
            trans_prob = _get_signal_at(trans_res.get(pair, {}), dt)

            if use_ep:
                # ── EP interne : pondérer par accuracy OOS ──
                # Plus le modèle est précis sur les derniers 63j, plus son vote pèse
                acc_gb    = gb_res.get(pair, {}).get('accuracy', 0.5)
                acc_lstm  = lstm_res.get(pair, {}).get('accuracy', 0.5)
                acc_trans = trans_res.get(pair, {}).get('accuracy', 0.5)
                # Poids = softmax des accuracy
                raw_w  = np.array([acc_gb, acc_trans, acc_lstm])
                raw_w  = np.exp((raw_w - 0.5) * 5.0)  # amplifier les différences
                raw_w /= raw_w.sum()
                blended = (raw_w[0] * (gb_prob - 0.5) * 2.0 +
                           raw_w[1] * (trans_prob - 0.5) * 2.0 +
                           raw_w[2] * (lstm_prob - 0.5) * 2.0)
            else:
                # ── Blend fixe original ──
                blended = (0.35 * (gb_prob - 0.5) * 2.0 +
                           0.40 * (trans_prob - 0.5) * 2.0 +
                           0.25 * (lstm_prob - 0.5) * 2.0)

            # Modulation HMM régime
            regime = "LOW-VOL"
            if pair in data_dict and 'Regime' in data_dict[pair]:
                avail = data_dict[pair]['Regime'].loc[:dt].dropna()
                if len(avail) > 0:
                    regime = avail.iloc[-1]
            boost  = {"LOW-VOL": 1.15, "HIGH-VOL": 0.80}.get(regime, 1.0)
            signals[i] = blended * boost

        if use_blmv and loc >= USER.lookback_cov:
            # ── BL/MV sur le signal MT ──
            window = ret.iloc[max(0, loc - USER.lookback_cov):loc]
            R      = window.to_numpy()
            cov    = np.cov(R.T) + 1e-6 * np.eye(n)
            mu_eq  = USER.risk_aversion * cov @ (np.ones(n) / n)
            P, Q   = _build_bl_views_from_ml(assets, gb_res, dt,
                                              n_views=5, view_scale=0.08)
            mu_bl  = compute_bl_posterior(mu_eq, cov, P, Q)
            bounds = [(-USER.max_weight, USER.max_weight)] * n
            cons   = [{'type': 'ineq',
                        'fun': lambda w: USER.max_leverage * lev_scale - np.sum(np.abs(w))}]
            result = minimize(
                lambda w: -(w @ mu_bl - USER.risk_aversion / 2 * w @ cov @ w),
                np.zeros(n), method='SLSQP', bounds=bounds, constraints=cons
            )
            w_new = result.x if result.success else np.tanh(signals * 1.5)
        else:
            w_new = np.tanh(signals * 1.5)

        # Normaliser
        max_lev = USER.max_leverage * lev_scale
        gross   = np.sum(np.abs(w_new))
        if gross > max_lev:
            w_new *= max_lev / gross
        w_new[np.abs(w_new) < 0.01] = 0.0

        turnover = np.sum(np.abs(w_new - w_prev))
        if turnover > 0.01:
            port_returns.loc[dt] -= turnover * USER.transaction_cost
        w_prev = w_new

    return StrategyResult(
        name=f"2. Moyen Terme ({pipeline['model']})",
        daily_returns=port_returns.dropna(),
        description=f"MT ensemble, EP={use_ep}, BL/MV={use_blmv}, lev={lev_scale:.0%}×max"
    ).compute_metrics()


# ════════════════════════════════════════════════════════════════════════════════
# LONG TERME — Carry + Momentum BL / RF BL / MLP macro BL
# ════════════════════════════════════════════════════════════════════════════════

def strategy_long_term(ret: pd.DataFrame, ret_clean: pd.DataFrame,
                        carry: pd.DataFrame, prices: pd.DataFrame,
                        lt_res: dict = None) -> StrategyResult:
    """
    Long terme (6 mois - 2 ans).

    Signaux utilisés :
    - Momentum 12 mois (en skippant le dernier mois) — Moskowitz, Ooi & Pedersen (2012)
    - Carry annualisé — Koijen et al. (2012)
    - Vues ML (RF 63j ou MLP macro 63j) si disponibles

    Rebalancement trimestriel (63j) ou mensuel selon profil.
    Covariance sur 252j (1 an) pour plus de stabilité.
    """
    pipeline   = USER.get_active_pipelines()['LT']
    model_id   = pipeline['model']
    use_ml_bl  = lt_res is not None and len(lt_res) > 0
    lev_scale  = pipeline['leverage']
    rebal_freq = pipeline['rebal_freq']   # 63j conservateur, 42j équilibré, 21j agressif

    hdr(f"LONG TERME — {model_id} | rebal/{rebal_freq}j | BL/MV={pipeline['bl_mv']}")

    assets       = ret.columns.tolist()
    n            = len(assets)
    dates        = ret.loc[BACKTEST_START:].index
    lookback_cov = USER.lookback_cov        # 252j (1 an)

    # Momentum 12m skip-1m : log(prix_t-21) - log(prix_t-252)
    # Convention standard CTA (Moskowitz 2012) — évite la mean-reversion CT
    log_prices    = np.log(prices[assets].ffill())
    momentum_12m  = (log_prices.shift(21) - log_prices.shift(USER.lookback_momentum_lt)).shift(1)
    # Momentum 3m pour les profils plus réactifs
    momentum_3m   = (log_prices - log_prices.shift(USER.lookback_momentum_mt)).shift(1)

    port_returns = pd.Series(0.0, index=dates, dtype=float)
    w_prev       = np.zeros(n)

    for t_idx, dt in enumerate(dates):
        loc = ret.index.get_loc(dt)
        port_returns.loc[dt] = float(w_prev @ ret.iloc[loc].values)

        if t_idx % rebal_freq != 0 or loc < lookback_cov:
            continue

        # Covariance 1 an sur rendements clean (plus robuste)
        window_clean = ret_clean.iloc[max(0, loc - lookback_cov):loc]
        if len(window_clean) < 100:
            continue
        cov   = window_clean.cov().values + 1e-6 * np.eye(n)
        mu_eq = USER.risk_aversion * cov @ (np.ones(n) / n)

        if use_ml_bl and model_id in ('rf_lt', 'mlp_macro_lt'):
            # ── Vues ML (RF ou MLP) → BL ──
            # Le modèle a prédit direction à 63j → vue calibrée
            P, Q = _build_bl_views_from_ml(assets, lt_res, dt,
                                            n_views=5, view_scale=0.10)
            mu_bl = compute_bl_posterior(mu_eq, cov, P, Q)

        else:
            # ── Vues classiques : carry + momentum 12m ──
            if dt not in momentum_12m.index:
                continue
            mom12 = momentum_12m.loc[dt].values
            mom3  = momentum_3m.loc[dt].values if dt in momentum_3m.index else np.zeros(n)
            carry_now = carry.loc[:dt].iloc[-1].values if len(carry.loc[:dt]) > 0 else np.zeros(n)

            if np.any(np.isnan(mom12)):
                continue

            # Normalisation cross-sectionnel — élimine effet TRY/ZAR
            mom12_cs  = (mom12 - np.nanmean(mom12)) / (np.nanstd(mom12) + 1e-10)
            mom3_cs   = (mom3  - np.nanmean(mom3))  / (np.nanstd(mom3)  + 1e-10)
            carry_ann = carry_now * TRADING_DAYS

            # Blend : 50% momentum 12m + 30% carry + 20% momentum 3m
            combined = 0.50 * mom12_cs + 0.30 * carry_ann + 0.20 * mom3_cs
            ranked   = np.argsort(combined)
            n_views  = min(5, max(3, n // 4))
            P, Q     = [], []
            for idx in ranked[-n_views:]:
                row = np.zeros(n); row[idx] = 1.0
                P.append(row); Q.append(np.tanh(combined[idx]) * 0.10)
            for idx in ranked[:n_views]:
                row = np.zeros(n); row[idx] = -1.0
                P.append(row); Q.append(np.tanh(combined[idx]) * 0.10)
            P, Q  = np.array(P), np.array(Q)
            mu_bl = compute_bl_posterior(mu_eq, cov, P, Q)

        # ── Optimisation Markowitz sur mu_bl ──
        max_lev = USER.max_leverage * lev_scale
        bounds  = [(-USER.max_weight * 0.8, USER.max_weight * 0.8)] * n
        cons    = [{'type': 'ineq', 'fun': lambda w: max_lev - np.sum(np.abs(w))}]

        result = minimize(
            lambda w: -(w @ mu_bl - USER.risk_aversion / 2 * w @ cov @ w),
            np.zeros(n), method='SLSQP', bounds=bounds, constraints=cons
        )
        w_new = result.x if result.success else np.zeros(n)

        # Vol targeting
        w_new = _vol_target_scale(w_new, cov, USER.target_vol * lev_scale, max_lev)
        w_new = np.clip(w_new, -USER.max_weight, USER.max_weight)

        turnover = np.sum(np.abs(w_new - w_prev))
        if turnover > 0.05:
            port_returns.loc[dt] -= turnover * USER.transaction_cost
        w_prev = w_new

    return StrategyResult(
        name=f"3. Long Terme ({model_id})",
        daily_returns=port_returns.dropna(),
        description=(f"LT BL+MV, {model_id}, mom12m+carry, "
                     f"lev={lev_scale:.0%}×max, rebal/{rebal_freq}j")
    ).compute_metrics()


# ════════════════════════════════════════════════════════════════════════════════
# ENTROPY POOLING STANDALONE
# ════════════════════════════════════════════════════════════════════════════════

def strategy_ep_standalone(ret: pd.DataFrame, data_dict: dict,
                             carry: pd.DataFrame, prices: pd.DataFrame,
                             gb_res: dict = None,
                             trans_res: dict = None) -> StrategyResult:
    """
    EP standalone — stratégie indépendante.

    Construit des vues directement à partir des signaux disponibles
    (carry, momentum CS, Hurst, Kalman/HMM, ML si disponible),
    applique Entropy Pooling pour obtenir la distribution posterior,
    puis CVaR pour le sizing.

    Modes selon ep_pipeline :
    - ep_signals  : vues brutes uniquement (sans ML)
    - ep_ml       : vues ML (GB + Transformer)
    - ep_ml_full  : toutes les vues pondérées par accuracy

    Ref :
      Meucci (2008) — Entropy Pooling
      Rockafellar & Uryasev (2000) — CVaR
      DMVFEP (2024)
    """
    pipeline   = USER.get_active_pipelines()['EP']
    model_id   = pipeline['model']
    use_ml     = model_id in ('ep_ml', 'ep_ml_full') and gb_res is not None
    lev_scale  = pipeline['leverage']

    hdr(f"ENTROPY POOLING STANDALONE — {model_id}")

    assets = ret.columns.tolist()
    n      = len(assets)
    dates  = ret.loc[BACKTEST_START:].index

    port_returns = pd.Series(0.0, index=dates, dtype=float)
    w_prev       = np.zeros(n)

    log_prices  = np.log(prices[assets].ffill())
    momentum_df = (log_prices - log_prices.shift(USER.lookback_momentum)).shift(1)

    for t_idx, dt in enumerate(dates):
        loc = ret.index.get_loc(dt)
        port_returns.loc[dt] = float(w_prev @ ret.iloc[loc].values)

        if t_idx % pipeline['rebal_freq'] != 0 or loc < USER.lookback_cov:
            continue

        window = ret.iloc[max(0, loc - USER.lookback_cov):loc]
        R      = window.to_numpy()
        if R.shape[0] < 50:
            continue

        mu_view = np.zeros(n)

        # ── Vues signaux bruts ──
        for i, a in enumerate(assets):
            pair = TICKER_TO_PAIR.get(a, a.replace("=X", ""))
            if pair not in data_dict:
                continue
            df    = data_dict[pair]
            avail = df.loc[:dt]
            if len(avail) < 50:
                continue

            # Carry
            carry_v = 0.0
            if a in carry.columns:
                c = carry[a].loc[:dt].dropna()
                if len(c) > 0:
                    carry_v = float(c.iloc[-1])

            # Momentum CS normalisé
            mom_cs = 0.0
            if 'Momentum_CS' in df.columns:
                m = avail['Momentum_CS'].dropna()
                if len(m) > 0:
                    mom_cs = float(m.iloc[-1])

            # Hurst signal
            hurst_signal = 0.0
            if 'Hurst' in df.columns and 'Hurst_Prediction' in df.columns:
                H  = avail['Hurst'].dropna()
                hp = avail['Hurst_Prediction'].dropna()
                if len(H) > 0 and len(hp) > 0:
                    h_val        = float(H.iloc[-1])
                    pred_val     = float(hp.iloc[-1])
                    hurst_signal = np.tanh(pred_val / 3.0) * abs(h_val - 0.5) * 2

            # HMM régime
            regime_adj = 1.0
            if 'Regime_Code' in df.columns:
                rc = avail['Regime_Code'].dropna()
                if len(rc) > 0:
                    regime_adj = 0.7 if int(rc.iloc[-1]) == 1 else 1.2   # high-vol → prudent

            # Blend des vues brutes
            mu_raw = (0.40 * carry_v / TRADING_DAYS +
                      0.35 * mom_cs * 0.001 / TRADING_DAYS +
                      0.25 * hurst_signal * 0.001 / TRADING_DAYS)

            if use_ml:
                # Ajouter la vue ML
                gb_prob     = _get_signal_at(gb_res.get(pair, {}), dt)
                trans_prob  = 0.5
                if trans_res is not None:
                    trans_prob = _get_signal_at(trans_res.get(pair, {}), dt)

                if model_id == 'ep_ml_full':
                    acc_gb    = gb_res.get(pair, {}).get('accuracy', 0.5)
                    acc_trans = trans_res.get(pair, {}).get('accuracy', 0.5) if trans_res else 0.5
                    w_gb      = max(0, acc_gb - 0.5) * 2
                    w_tr      = max(0, acc_trans - 0.5) * 2
                    total_w   = w_gb + w_tr + 1e-10
                    ml_sig    = (w_gb * (gb_prob - 0.5) + w_tr * (trans_prob - 0.5)) / total_w
                else:
                    ml_sig = 0.5 * ((gb_prob - 0.5) + (trans_prob - 0.5))

                # EP_ML : 50% vues brutes + 50% ML
                mu_view[i] = (0.5 * mu_raw + 0.5 * ml_sig * 0.002 / TRADING_DAYS) * regime_adj
            else:
                mu_view[i] = mu_raw * regime_adj

        # Borner mu_view
        mu_view = np.clip(mu_view, R.min(axis=0), R.max(axis=0))

        # ── Entropy Pooling → distribution posterior ──
        q, converged = entropy_pooling(R, mu_view)
        mu_post = R.T @ q

        # ── CVaR optimization ──
        max_lev = USER.max_leverage * lev_scale
        w_new   = cvar_optimize(R, mu_post,
                                 alpha=0.05,
                                 max_weight=USER.max_weight,
                                 max_leverage=max_lev,
                                 risk_aversion=USER.risk_aversion)

        # Vol targeting
        X       = R - mu_post
        cov_ep  = (X.T * q) @ X + 1e-6 * np.eye(n)
        w_new   = _vol_target_scale(w_new, cov_ep,
                                     USER.target_vol * lev_scale, max_lev)
        w_new   = np.clip(w_new, -USER.max_weight, USER.max_weight)

        turnover = np.sum(np.abs(w_new - w_prev))
        if turnover > 0.01:
            port_returns.loc[dt] -= turnover * USER.transaction_cost
        w_prev = w_new

    return StrategyResult(
        name=f"5. EP Standalone ({model_id})",
        daily_returns=port_returns.dropna(),
        description=f"EP+CVaR, {model_id}, lev={lev_scale:.0%}×max, rebal/{pipeline['rebal_freq']}j"
    ).compute_metrics()


# ════════════════════════════════════════════════════════════════════════════════
# COMBINÉ — Entropy Pooling fusion CT + MT + LT + Vol Targeting
# ════════════════════════════════════════════════════════════════════════════════

def strategy_combined(ret: pd.DataFrame, ret_clean: pd.DataFrame,
                       carry: pd.DataFrame, prices: pd.DataFrame,
                       data_dict: dict,
                       gb_res: dict, lstm_res: dict, trans_res: dict,
                       ct_res: dict = None, lt_res: dict = None) -> StrategyResult:
    """
    Stratégie combinée améliorée :
    1. Vues CT (ML ou z-score)
    2. Vues MT (ensemble ML, EP calibré si profil agressif)
    3. Vues LT (RF/MLP ou carry+mom)
    4. Fusion via Entropy Pooling pondérée par USER.horizon_weights
    5. Vol Targeting + contraintes

    Ref : Meucci (2008), DMVFEP (2024).
    """
    pipeline  = USER.get_active_pipelines()['COMBINED']
    lev_scale = pipeline['leverage']

    hdr("COMBINÉ — Entropy Pooling fusion CT+MT+LT + Vol Targeting")

    assets = ret.columns.tolist()
    n      = len(assets)
    dates  = ret.loc[BACKTEST_START:].index
    hw     = USER.horizon_weights

    port_returns = pd.Series(0.0, index=dates, dtype=float)
    w_prev       = np.zeros(n)
    all_weights  = []

    log_prices   = np.log(prices[assets].ffill())
    # Momentum 12m skip-1m pour LT (Moskowitz 2012)
    momentum_12m = (log_prices.shift(21) - log_prices.shift(USER.lookback_momentum_lt)).shift(1)
    # Momentum 3m pour MT
    momentum_3m  = (log_prices - log_prices.shift(USER.lookback_momentum_mt)).shift(1)

    # EP standalone : rebal selon pipeline EP
    ep_rebal = USER.get_active_pipelines()['EP']['rebal_freq']

    for t_idx, dt in enumerate(dates):
        loc = ret.index.get_loc(dt)
        port_returns.loc[dt] = float(w_prev @ ret.iloc[loc].values)

        if t_idx % 5 != 0 or loc < USER.lookback_cov:
            continue

        window = ret.iloc[max(0, loc - USER.lookback_cov):loc]
        R      = window.to_numpy()
        if R.shape[0] < 50:
            continue

        mu_prior = R.mean(axis=0)

        # ── Vues COURT TERME ──
        mu_ct = np.zeros(n)
        for i, a in enumerate(assets):
            pair = TICKER_TO_PAIR.get(a, a.replace("=X", ""))
            if pair not in data_dict:
                continue
            df    = data_dict[pair]
            avail = df.loc[:dt]
            if len(avail) < 50:
                continue

            if ct_res and pair in ct_res:
                prob     = _get_signal_at(ct_res[pair], dt)
                mu_ct[i] = (prob - 0.5) * 2.0 * 0.001 / TRADING_DAYS
            else:
                H, pred  = 0.5, 0.0
                if 'Hurst' in df.columns:
                    h = avail['Hurst'].dropna()
                    if len(h) > 0: H = float(h.iloc[-1])
                if 'Hurst_Prediction' in df.columns:
                    hp = avail['Hurst_Prediction'].dropna()
                    if len(hp) > 0: pred = float(hp.iloc[-1])
                if H < 0.45:
                    ret_1d = avail['Return_1d'].dropna()
                    z = ret_1d.iloc[-21:].mean() / (ret_1d.iloc[-21:].std() + 1e-10) if len(ret_1d) >= 21 else 0
                    mu_ct[i] = (-np.tanh(z) * (0.5 - H) + pred / 100) / TRADING_DAYS

        # ── Vues MOYEN TERME ──
        mu_mt = np.zeros(n)
        for i, a in enumerate(assets):
            pair = TICKER_TO_PAIR.get(a, a.replace("=X", ""))
            gb_prob    = _get_signal_at(gb_res.get(pair, {}),    dt)
            lstm_prob  = _get_signal_at(lstm_res.get(pair, {}),  dt)
            trans_prob = _get_signal_at(trans_res.get(pair, {}), dt)

            # EP calibré si agressif
            if USER.mt_pipeline == 'agressif':
                acc_gb    = gb_res.get(pair, {}).get('accuracy', 0.5)
                acc_lstm  = lstm_res.get(pair, {}).get('accuracy', 0.5)
                acc_trans = trans_res.get(pair, {}).get('accuracy', 0.5)
                raw_w     = np.exp(np.array([acc_gb, acc_trans, acc_lstm]) * 5.0)
                raw_w    /= raw_w.sum()
                blended   = (raw_w[0] * (gb_prob - 0.5) * 2.0 +
                             raw_w[1] * (trans_prob - 0.5) * 2.0 +
                             raw_w[2] * (lstm_prob - 0.5) * 2.0)
            else:
                blended = (0.35 * (gb_prob - 0.5) * 2.0 +
                           0.40 * (trans_prob - 0.5) * 2.0 +
                           0.25 * (lstm_prob - 0.5) * 2.0)
            mu_mt[i] = blended * 0.001 / TRADING_DAYS

        # ── Vues LONG TERME ──
        mu_lt = np.zeros(n)
        if lt_res and len(lt_res) > 0:
            for i, a in enumerate(assets):
                pair = TICKER_TO_PAIR.get(a, a.replace("=X", ""))
                prob     = _get_signal_at(lt_res.get(pair, {}), dt)
                mu_lt[i] = (prob - 0.5) * 2.0 * 0.001 / TRADING_DAYS
        elif dt in momentum_12m.index:
            mom12 = momentum_12m.loc[dt].values
            carry_now = carry.loc[:dt].iloc[-1].values if len(carry.loc[:dt]) > 0 else np.zeros(n)
            if not np.any(np.isnan(mom12)):
                mom12_cs = (mom12 - np.nanmean(mom12)) / (np.nanstd(mom12) + 1e-10)
                carry_ann = carry_now * TRADING_DAYS
                mu_lt = 0.60 * mom12_cs / TRADING_DAYS + 0.40 * carry_now

        # ── Fusion pondérée ──
        mu_view = (hw['short'] * mu_ct +
                   hw['medium'] * mu_mt +
                   hw['long'] * mu_lt)
        mu_view = np.clip(mu_view, R.min(axis=0), R.max(axis=0))

        # ── Entropy Pooling ──
        q, converged = entropy_pooling(R, mu_view)
        mu_post = R.T @ q
        X       = R - mu_post
        Sigma   = (X.T * q) @ X + 1e-6 * np.eye(n)

        # ── Portefeuille tangent ──
        w = np.linalg.pinv(Sigma) @ mu_post
        w = np.clip(w, -USER.max_weight, USER.max_weight)
        gross = np.sum(np.abs(w))
        if gross > USER.max_leverage:
            w *= USER.max_leverage / gross

        # ── Vol Targeting ──
        max_lev = USER.max_leverage * lev_scale
        w       = _vol_target_scale(w, Sigma, USER.target_vol, max_lev)
        w       = np.clip(w, -USER.max_weight, USER.max_weight)

        turnover = np.sum(np.abs(w - w_prev))
        port_returns.loc[dt] -= turnover * USER.transaction_cost
        w_prev = w
        all_weights.append([dt] + w.tolist())

    weights_df = (pd.DataFrame(all_weights, columns=["Date"] + assets).set_index("Date")
                  if all_weights else pd.DataFrame())

    return StrategyResult(
        name="4. Combiné (EP + 3 horizons)",
        daily_returns=port_returns.dropna(),
        weights_history=weights_df,
        description=f"EP fusion CT({hw['short']:.0%})+MT({hw['medium']:.0%})+LT({hw['long']:.0%}), "
                    f"vol target {USER.target_vol:.0%}"
    ).compute_metrics()