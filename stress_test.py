#!/usr/bin/env python3
"""
STRESS TEST — Vérifie si le Sharpe CT=5.9 est réel ou artefact.

Tests :
  1. Spread multiplier (1x, 2x, 3x, 5x)
  2. Périodes différentes (2015-2020 vs 2020-2025)
  3. Signal randomisé (baseline : si random fait aussi bien → signal bidon)

Usage :
  python stress_test.py
  (fx_framework.py + CSVs doivent être dans le même dossier)
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from copy import deepcopy

warnings.filterwarnings('ignore')
np.random.seed(42)

# Import depuis le framework principal
from fx_framework import (
    USER, BACKTEST_START, TRADING_DAYS, OUTPUT_DIR,
    SPECIFIC_SPREADS, DEFAULT_SPREAD,
    FEATURE_COLS_CT, FEATURE_COLS_MT,
    load_data, get_fx_universe, compute_returns, compute_carry,
    ticker_to_pair, get_spread,
    apply_wavelet, create_features, apply_hurst, apply_regimes, apply_garch,
    train_ridge_ct, train_lgbm_ct, train_gb_mt, train_transformer_mt,
    strategy_equal_weight, strategy_ct, strategy_mt, strategy_combined,
    StrategyResult, hdr, sub, tbl, _get_signal,
    _spread_cost_for_pair,
)


# ════════════════════════════════════════════════════════════════════════════════
# TEST 1 : SPREAD MULTIPLIER
# ════════════════════════════════════════════════════════════════════════════════

def strategy_ct_with_spread_mult(ret, data_dict, ridge_res, lgbm_res, spread_mult):
    """CT identique mais avec spread × multiplicateur."""
    w_ridge = USER.ct_model_weights['ridge']
    w_lgbm  = USER.ct_model_weights['lgbm']

    assets = ret.columns.tolist()
    n      = len(assets)
    dates  = ret.loc[BACKTEST_START:].index
    rebal  = USER.ct_rebal

    port_ret = pd.Series(0.0, index=dates, dtype=float)
    w_prev   = np.zeros(n)

    for t_idx, dt in enumerate(dates):
        loc = ret.index.get_loc(dt)
        port_ret.loc[dt] = float(w_prev @ ret.iloc[loc].values)

        if t_idx % rebal != 0:
            continue

        w_new = np.zeros(n)
        for i, a in enumerate(assets):
            pair = ticker_to_pair(a)
            if pair not in data_dict:
                continue
            df = data_dict[pair]
            if len(df.loc[:dt]) < 63:
                continue

            p_ridge = _get_signal(ridge_res.get(pair, {}), dt)
            p_lgbm  = _get_signal(lgbm_res.get(pair, {}), dt)
            blended = w_ridge * (p_ridge - 0.5) * 2 + w_lgbm * (p_lgbm - 0.5) * 2

            if 'Regime' in df.columns:
                reg = df['Regime'].loc[:dt].dropna()
                if len(reg) > 0:
                    boost = {"LOW-VOL": 1.1, "HIGH-VOL": 0.75}.get(reg.iloc[-1], 1.0)
                    blended *= boost

            w_new[i] = np.tanh(blended * 1.5)

        gross = np.sum(np.abs(w_new))
        if gross > USER.max_leverage * 0.5:
            w_new *= USER.max_leverage * 0.5 / gross

        # Spread × multiplier
        for i, a in enumerate(assets):
            pair = ticker_to_pair(a)
            turnover_i = abs(w_new[i] - w_prev[i])
            if turnover_i > 0.01:
                cost = _spread_cost_for_pair(pair, data_dict, dt) * spread_mult
                port_ret.loc[dt] -= turnover_i * cost

        w_prev = w_new

    return StrategyResult(
        name=f"CT spread×{spread_mult}",
        daily_returns=port_ret.dropna(),
    ).compute()


# ════════════════════════════════════════════════════════════════════════════════
# TEST 2 : PÉRIODE DIFFÉRENTE
# ════════════════════════════════════════════════════════════════════════════════

def strategy_ct_custom_period(ret, data_dict, ridge_res, lgbm_res, start_date):
    """CT avec une date de début différente."""
    w_ridge = USER.ct_model_weights['ridge']
    w_lgbm  = USER.ct_model_weights['lgbm']

    assets = ret.columns.tolist()
    n      = len(assets)

    # Filtrer les dates après start_date
    valid_dates = ret.loc[start_date:].index
    if len(valid_dates) < 50:
        return StrategyResult(name=f"CT from {start_date}", daily_returns=pd.Series(dtype=float)).compute()

    rebal = USER.ct_rebal
    port_ret = pd.Series(0.0, index=valid_dates, dtype=float)
    w_prev   = np.zeros(n)

    for t_idx, dt in enumerate(valid_dates):
        loc = ret.index.get_loc(dt)
        port_ret.loc[dt] = float(w_prev @ ret.iloc[loc].values)

        if t_idx % rebal != 0:
            continue

        w_new = np.zeros(n)
        for i, a in enumerate(assets):
            pair = ticker_to_pair(a)
            if pair not in data_dict:
                continue
            df = data_dict[pair]
            if len(df.loc[:dt]) < 63:
                continue

            p_ridge = _get_signal(ridge_res.get(pair, {}), dt)
            p_lgbm  = _get_signal(lgbm_res.get(pair, {}), dt)
            blended = w_ridge * (p_ridge - 0.5) * 2 + w_lgbm * (p_lgbm - 0.5) * 2

            if 'Regime' in df.columns:
                reg = df['Regime'].loc[:dt].dropna()
                if len(reg) > 0:
                    boost = {"LOW-VOL": 1.1, "HIGH-VOL": 0.75}.get(reg.iloc[-1], 1.0)
                    blended *= boost

            w_new[i] = np.tanh(blended * 1.5)

        gross = np.sum(np.abs(w_new))
        if gross > USER.max_leverage * 0.5:
            w_new *= USER.max_leverage * 0.5 / gross

        for i, a in enumerate(assets):
            pair = ticker_to_pair(a)
            turnover_i = abs(w_new[i] - w_prev[i])
            if turnover_i > 0.01:
                cost = _spread_cost_for_pair(pair, data_dict, dt)
                port_ret.loc[dt] -= turnover_i * cost

        w_prev = w_new

    return StrategyResult(
        name=f"CT ({start_date}→fin)",
        daily_returns=port_ret.dropna(),
    ).compute()


# ════════════════════════════════════════════════════════════════════════════════
# TEST 3 : SIGNAL RANDOMISÉ (null hypothesis)
# ════════════════════════════════════════════════════════════════════════════════

def strategy_ct_random_signals(ret, data_dict, n_trials=20):
    """
    Remplace les signaux ML par du random uniforme [0,1].
    Si le vrai CT n'est pas significativement meilleur → signal bidon.
    """
    assets = ret.columns.tolist()
    n      = len(assets)
    dates  = ret.loc[BACKTEST_START:].index
    rebal  = USER.ct_rebal

    sharpes = []

    for trial in range(n_trials):
        rng = np.random.RandomState(trial)
        port_ret = pd.Series(0.0, index=dates, dtype=float)
        w_prev   = np.zeros(n)

        for t_idx, dt in enumerate(dates):
            loc = ret.index.get_loc(dt)
            port_ret.loc[dt] = float(w_prev @ ret.iloc[loc].values)

            if t_idx % rebal != 0:
                continue

            # Signal random
            w_new = np.zeros(n)
            for i, a in enumerate(assets):
                pair = ticker_to_pair(a)
                if pair not in data_dict:
                    continue
                signal = (rng.random() - 0.5) * 2  # uniforme [-1, 1]
                w_new[i] = np.tanh(signal * 1.5)

            gross = np.sum(np.abs(w_new))
            if gross > USER.max_leverage * 0.5:
                w_new *= USER.max_leverage * 0.5 / gross

            for i, a in enumerate(assets):
                pair = ticker_to_pair(a)
                turnover_i = abs(w_new[i] - w_prev[i])
                if turnover_i > 0.01:
                    cost = _spread_cost_for_pair(pair, data_dict, dt)
                    port_ret.loc[dt] -= turnover_i * cost

            w_prev = w_new

        r = port_ret.dropna()
        if len(r) > 10 and r.std() > 0:
            sharpes.append(float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)))
        else:
            sharpes.append(0.0)

    return np.array(sharpes)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    # ── Setup (profil agressif pour reproduire le test) ──
    USER.set_profile('agressif')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Chargement (identique au framework) ──
    hdr("CHARGEMENT")
    prices, rates = load_data()
    tickers = get_fx_universe(prices, rates)
    pairs   = [ticker_to_pair(t) for t in tickers]
    print(f"  Univers : {len(pairs)} paires")
    ret   = compute_returns(prices[tickers])
    carry = compute_carry(tickers, rates).loc[ret.index[0]:]
    carry = carry.reindex(ret.index, method='ffill')

    data_dict = {}
    for t in tickers:
        pair = ticker_to_pair(t)
        df = pd.DataFrame({'Close': prices[t]}).dropna()
        df['Return_1d'] = df['Close'].pct_change()
        data_dict[pair] = df

    # ── Signal processing ──
    data_dict = apply_wavelet(data_dict)
    data_dict = create_features(data_dict)
    data_dict = apply_hurst(data_dict)
    data_dict = apply_regimes(data_dict)
    data_dict = apply_garch(data_dict)

    # ── ML (on réentraîne Ridge + LGBM, pas le Transformer pour aller vite) ──
    hdr("ML — Ridge + LGBM CT seulement (stress test)")
    ridge_res = train_ridge_ct(data_dict, pairs)
    lgbm_res  = train_lgbm_ct(data_dict, pairs)

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 1 : SPREAD MULTIPLIER
    # ══════════════════════════════════════════════════════════════════════════
    hdr("TEST 1 — SPREAD MULTIPLIER")
    print("  Si le Sharpe s'effondre à 2x-3x spread → signal fragile\n")

    spread_results = []
    for mult in [1.0, 2.0, 3.0, 5.0, 10.0]:
        r = strategy_ct_with_spread_mult(ret, data_dict, ridge_res, lgbm_res, mult)
        spread_results.append([f"×{mult:.0f}", f"{r.sharpe:.3f}", f"{r.cagr:.2%}",
                               f"{r.max_dd:.2%}", f"{r.ann_vol:.2%}", f"{r.win_rate:.1%}"])

    tbl(["Spread", "Sharpe", "CAGR", "MaxDD", "Vol", "Win%"], spread_results)

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 2 : SOUS-PÉRIODES
    # ══════════════════════════════════════════════════════════════════════════
    hdr("TEST 2 — SOUS-PÉRIODES")
    print("  Si le Sharpe varie énormément entre périodes → overfitting\n")

    period_results = []
    for start in ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"]:
        r = strategy_ct_custom_period(ret, data_dict, ridge_res, lgbm_res, start)
        n_days = len(r.daily_returns) if r.daily_returns is not None else 0
        period_results.append([f"{start}→fin", str(n_days) + "j",
                               f"{r.sharpe:.3f}", f"{r.cagr:.2%}",
                               f"{r.max_dd:.2%}", f"{r.win_rate:.1%}"])

    tbl(["Période", "Durée", "Sharpe", "CAGR", "MaxDD", "Win%"], period_results)

    # ══════════════════════════════════════════════════════════════════════════
    # TEST 3 : SIGNAL RANDOM (null hypothesis)
    # ══════════════════════════════════════════════════════════════════════════
    hdr("TEST 3 — SIGNAL RANDOM (20 trials)")
    print("  Null hypothesis : un signal random fait-il aussi bien ?\n")

    random_sharpes = strategy_ct_random_signals(ret, data_dict, n_trials=20)

    # Sharpe réel
    real_ct = strategy_ct_with_spread_mult(ret, data_dict, ridge_res, lgbm_res, 1.0)

    print(f"  Sharpe CT réel        : {real_ct.sharpe:.3f}")
    print(f"  Sharpe random moyen   : {np.mean(random_sharpes):.3f}")
    print(f"  Sharpe random max     : {np.max(random_sharpes):.3f}")
    print(f"  Sharpe random std     : {np.std(random_sharpes):.3f}")
    print(f"  Sharpe random p95     : {np.percentile(random_sharpes, 95):.3f}")
    print()

    # Z-score du vrai signal vs distribution random
    z = (real_ct.sharpe - np.mean(random_sharpes)) / (np.std(random_sharpes) + 1e-10)
    print(f"  Z-score (réel vs random) : {z:.1f}")
    if z > 3:
        print("  → Signal SIGNIFICATIF (z > 3)")
    elif z > 2:
        print("  → Signal probablement réel (z > 2)")
    elif z > 1:
        print("  → Signal faible, possiblement du bruit (z > 1)")
    else:
        print("  → Signal PAS significatif (z ≤ 1) — probablement du bruit")

    # ══════════════════════════════════════════════════════════════════════════
    # RÉSUMÉ
    # ══════════════════════════════════════════════════════════════════════════
    hdr("VERDICT")
    print(f"""
  Sharpe original (spread 1x) : {real_ct.sharpe:.3f}
  Sharpe spread 3x            : {spread_results[2][1]}
  Sharpe spread 5x            : {spread_results[3][1]}
  Sharpe random max (20 runs) : {np.max(random_sharpes):.3f}
  Z-score vs random           : {z:.1f}

  INTERPRÉTATION :
  • Si Sharpe 3x > 2.0 ET z > 3 → signal CT est RÉEL et ROBUSTE
  • Si Sharpe 3x > 1.0 ET z > 2 → signal existe mais fragile
  • Si Sharpe 3x < 1.0 OU z < 2 → probablement artefact / overfitting
    """)


if __name__ == '__main__':
    main()