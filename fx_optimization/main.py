#!/usr/bin/env python3
"""
FX PORTFOLIO OPTIMIZATION v4
"""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np

from config import (
    BACKTEST_START, PAIR_MAP, TICKER_TO_PAIR, USER, OUTPUT_DIR,
    PIPELINE_REGISTRY, print_pipeline_recommendations
)
from data_utils import load_data, get_fx_universe, compute_returns, compute_carry, ticker_to_pair_name
from signal_processing import apply_wavelet_denoising, apply_hurst, apply_regimes, apply_garch
from feature_engineering import create_features, compute_ic
from ml_models import (
    train_gb_walkforward, train_lstm_walkforward, train_transformer_walkforward,
    train_ridge_ct_walkforward, train_lgbm_ct_walkforward,
    train_rf_lt_walkforward, train_mlp_macro_lt_walkforward
)
from strategies import (
    strategy_equal_weight, strategy_short_term, strategy_medium_term,
    strategy_long_term, strategy_combined, strategy_ep_standalone
)
from utils_visualization import plot_equity_curves, plot_comparison, plot_signals
from utils_display import hdr, sub, tbl
from metrics_performance import print_comparison_table, summary_to_dataframe


def interactive_setup():
    print("\n" + "=" * 80)
    print("  FX PORTFOLIO OPTIMIZATION v4 — Configuration")
    print("=" * 80)

    print("\n  [1/4] PROFIL DE RISQUE")
    print("    [1] Conservateur  — vol=5%,  levier=1x")
    print("    [2] Equilibre     — vol=10%, levier=1.5x")
    print("    [3] Agressif      — vol=18%, levier=2x")
    print("    [4] Personnalise")
    choice = input("\n  Choix (1-4) [2] : ").strip() or '2'

    if choice == '1':
        USER.set_profile('conservateur'); profil_label = 'conservateur'
    elif choice == '3':
        USER.set_profile('agressif'); profil_label = 'agressif'
    elif choice == '4':
        profil_label = 'equilibre'; USER.set_profile('equilibre')
        try:
            v = input("    Vol cible (ex: 0.10) : ").strip()
            if v: USER.target_vol = float(v)
            v = input("    Levier max (ex: 1.5) : ").strip()
            if v: USER.max_leverage = float(v)
            v = input("    Aversion risque 1-5 : ").strip()
            if v: USER.risk_aversion = float(v)
        except ValueError:
            USER.set_profile('equilibre')
    else:
        USER.set_profile('equilibre'); profil_label = 'equilibre'

    print_pipeline_recommendations(profil_label)

    print("\n  [2/4] PIPELINES PAR HORIZON (ENTER = auto selon profil)")
    for h, attr in [('CT','ct_pipeline'),('MT','mt_pipeline'),('LT','lt_pipeline'),('EP','ep_pipeline')]:
        options = list(PIPELINE_REGISTRY[h].keys())
        default = getattr(USER, attr)
        print(f"  {h}: {' / '.join(options)}  [defaut={default}]")
        v = input(f"    {h} -> ").strip().lower()
        if v and v in options:
            setattr(USER, attr, v)

    print("\n  [3/4] PAIRES")
    all_pairs = sorted(PAIR_MAP.keys())
    for i, p in enumerate(all_pairs, 1):
        print(f"    [{i:2d}] {p}", end="   ")
        if i % 5 == 0: print()
    print()
    sel = input("\n  Paires (ex: 1,2,5 ou ENTER=toutes) : ").strip()
    if sel:
        try:
            indices = [int(x.strip()) - 1 for x in sel.split(',')]
            USER.selected_pairs = [all_pairs[i] for i in indices if 0 <= i < len(all_pairs)]
        except (ValueError, IndexError):
            USER.selected_pairs = None
    else:
        USER.selected_pairs = None

    print("\n  [4/4] MODE")
    print("    [1] Backtest complet (CT+MT+LT+EP+Combine)")
    print("    [2] Backtest rapide  (MT+Combine uniquement)")
    mode = input("  Choix [1] : ").strip() or '1'

    v = input("  IC scoring des features ? (O/n) [O] : ").strip().lower()
    run_ic = (v != 'n')

    print("\n" + "-" * 60)
    print(USER.summary())
    confirm = input("\n  Confirmer ? (O/n) : ").strip().lower()
    if confirm == 'n':
        return interactive_setup()
    return mode, run_ic


def main():
    run_ic = False
    if '--default' in sys.argv:
        mode = '1'; USER.set_profile('equilibre')
    else:
        mode, run_ic = interactive_setup()

    # ETAPE 1 : CHARGEMENT
    hdr("CHARGEMENT DES DONNEES")
    prices_fx, rates = load_data()
    active_tickers   = get_fx_universe(prices_fx, rates)
    pairs            = [ticker_to_pair_name(t) for t in active_tickers]
    print(f"  Univers : {len(pairs)} paires")
    ret   = compute_returns(prices_fx[active_tickers])
    carry = compute_carry(active_tickers, rates).loc[ret.index[0]:]
    carry = carry.reindex(ret.index, method='ffill')

    # ETAPE 2 : DATA_DICT
    hdr("PREPARATION DATA_DICT")
    data_dict = {}
    for t in active_tickers:
        pair = ticker_to_pair_name(t)
        df   = pd.DataFrame({'Close': prices_fx[t]}).dropna()
        df['Return_1d'] = df['Close'].pct_change()
        data_dict[pair] = df

    # ETAPE 3 : SIGNAL (Wavelet — Kalman supprime)
    data_dict = apply_wavelet_denoising(data_dict)

    # ETAPE 4 : FEATURES
    data_dict = create_features(data_dict)

    # ETAPE 5 : HURST + HMM + GARCH
    hdr("ANALYSE AVANCEE")
    data_dict = apply_hurst(data_dict)
    data_dict = apply_regimes(data_dict)
    data_dict = apply_garch(data_dict)

    # ETAPE 6 : IC SCORING (optionnel)
    if run_ic:
        from config import FEATURE_COLS_MT
        ic_df = compute_ic(data_dict, feature_cols=FEATURE_COLS_MT)
        ic_df.to_csv(os.path.join(OUTPUT_DIR, "ic_scores.csv"), index=False)

    # ETAPE 7 : RENDEMENTS CLEAN
    ret_clean = pd.DataFrame({
        t: pd.Series(data_dict[ticker_to_pair_name(t)].get(
            'Return_Clean', data_dict[ticker_to_pair_name(t)]['Return_1d']))
        for t in active_tickers
    }).dropna()
    ret_clean = ret_clean.loc[ret.index.intersection(ret_clean.index)]

    # ETAPE 8 : ML
    ct_res = lt_res = {}
    gb_res = lstm_res = trans_res = {}

    hdr(f"ENTRAINEMENT ML (walk-forward, retrain/{USER.wf_retrain_freq}j)")

    if mode == '1':
        sub(f"Court Terme : {USER.ct_pipeline}")
        if USER.ct_pipeline == 'conservateur':
            ct_res = train_ridge_ct_walkforward(data_dict, pairs)
        elif USER.ct_pipeline == 'agressif':
            ct_res = train_lgbm_ct_walkforward(data_dict, pairs, use_hurst_filter=True)
        else:
            ct_res = train_lgbm_ct_walkforward(data_dict, pairs, use_hurst_filter=False)

    sub(f"Moyen Terme : {USER.mt_pipeline}")
    gb_res    = train_gb_walkforward(data_dict, pairs)
    lstm_res  = train_lstm_walkforward(data_dict, pairs)
    trans_res = train_transformer_walkforward(data_dict, pairs)

    if mode == '1':
        sub(f"Long Terme : {USER.lt_pipeline}")
        if USER.lt_pipeline == 'equilibre':
            lt_res = train_rf_lt_walkforward(data_dict, pairs)
        elif USER.lt_pipeline == 'agressif':
            lt_res = train_mlp_macro_lt_walkforward(data_dict, pairs)

    # ETAPE 9 : STRATEGIES
    hdr(f"EXECUTION STRATEGIES ({BACKTEST_START} -> aujourd'hui)")
    all_results = []

    s0 = strategy_equal_weight(ret.loc[BACKTEST_START:])
    all_results.append(s0)
    print(f"  [0] {s0.name}: Sharpe={s0.sharpe:.3f}  CAGR={s0.cagr:.2%}")

    if mode == '1':
        s1 = strategy_short_term(ret, data_dict, ct_res=ct_res or None)
        all_results.append(s1)
        print(f"  [1] {s1.name}: Sharpe={s1.sharpe:.3f}  CAGR={s1.cagr:.2%}")

    s2 = strategy_medium_term(ret, data_dict, gb_res, lstm_res, trans_res)
    all_results.append(s2)
    print(f"  [2] {s2.name}: Sharpe={s2.sharpe:.3f}  CAGR={s2.cagr:.2%}")

    if mode == '1':
        s3 = strategy_long_term(ret, ret_clean, carry, prices_fx, lt_res=lt_res or None)
        all_results.append(s3)
        print(f"  [3] {s3.name}: Sharpe={s3.sharpe:.3f}  CAGR={s3.cagr:.2%}")

        s5 = strategy_ep_standalone(ret, data_dict, carry, prices_fx,
                                     gb_res=gb_res, trans_res=trans_res)
        all_results.append(s5)
        print(f"  [5] {s5.name}: Sharpe={s5.sharpe:.3f}  CAGR={s5.cagr:.2%}")

    s4 = strategy_combined(ret, ret_clean, carry, prices_fx, data_dict,
                            gb_res, lstm_res, trans_res,
                            ct_res=ct_res or None, lt_res=lt_res or None)
    all_results.append(s4)
    print(f"  [4] {s4.name}: Sharpe={s4.sharpe:.3f}  CAGR={s4.cagr:.2%}")

    # ETAPE 10 : COMPARAISON + SAUVEGARDE
    print_comparison_table(all_results)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_equity_curves(all_results)
    plot_comparison(all_results)
    plot_signals(data_dict)
    summary_to_dataframe(all_results).to_csv(
        os.path.join(OUTPUT_DIR, "strategy_comparison.csv"), index=False)

    print("\n" + "=" * 80)
    if all_results:
        best = max(all_results[1:], key=lambda r: r.sharpe)
        print(f"  Meilleure : {best.name} (Sharpe={best.sharpe:.3f}, CAGR={best.cagr:.2%})")
    print(f"  Outputs   : {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()