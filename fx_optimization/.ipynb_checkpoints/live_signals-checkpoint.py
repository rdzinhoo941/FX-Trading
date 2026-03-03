"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  SIGNAUX LIVE — "Quoi trader aujourd'hui ?"                                    ║
║                                                                                ║
║  Génère des signaux exploitables pour chaque paire, par horizon :             ║
║    • Court terme : mean-reversion (Hurst + Kalman + Garcin)                   ║
║    • Moyen terme : ML ensemble (LSTM + Transformer + GB)                       ║
║    • Long terme  : carry + momentum + régime                                   ║
║    • Combiné     : signal pondéré + sizing                                    ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from config import USER, TICKER_TO_PAIR, TRADING_DAYS
from utils_display import hdr, tbl


def generate_live_signals(data_dict: dict, pairs: list,
                          carry: pd.DataFrame, prices: pd.DataFrame,
                          gb_res: dict, lstm_res: dict, trans_res: dict) -> pd.DataFrame:
    """
    Génère le tableau de signaux pour aujourd'hui.
    Retourne un DataFrame avec un signal par paire et par horizon.
    """
    hdr("SIGNAUX LIVE — Recommandations du jour")
    hw = USER.horizon_weights
    print(f"  Profil : vol cible={USER.target_vol:.0%}, "
          f"levier max={USER.max_leverage:.1f}x, "
          f"aversion={USER.risk_aversion}")
    print(f"  Horizons : CT={hw['short']:.0%}  MT={hw['medium']:.0%}  LT={hw['long']:.0%}")
    print()

    signals = []

    for pair in pairs:
        if pair not in data_dict:
            continue
        df = data_dict[pair]
        if len(df) < 50:
            continue

        # ── COURT TERME ──
        H = df['Hurst'].dropna().iloc[-1] if 'Hurst' in df and not df['Hurst'].dropna().empty else 0.5
        kalman = df['Kalman_Residual'].iloc[-1] if 'Kalman_Residual' in df else 0
        rsi = df['RSI'].iloc[-1] if 'RSI' in df and not pd.isna(df['RSI'].iloc[-1]) else 0.5
        garcin = df['Hurst_Prediction'].dropna().iloc[-1] if 'Hurst_Prediction' in df and not df['Hurst_Prediction'].dropna().empty else 0

        ct_signal = 0.0
        if H < 0.45:
            strength = 2 * (0.5 - H)
            ct_signal = (-np.tanh(kalman) * strength * 0.30 +
                         garcin / 100 * 0.50 +
                         (0.5 - rsi) * strength * 0.20 if rsi < 0.30 or rsi > 0.70 else 0)

        # ── MOYEN TERME ──
        l = lstm_res.get(pair, {}).get('expected_return_bps', 0)
        t = trans_res.get(pair, {}).get('expected_return_bps', 0)
        g = gb_res.get(pair, {}).get('expected_return_bps', 0)
        mt_signal = np.tanh((0.25 * l + 0.40 * t + 0.35 * g) / 30)

        # Modulation par régime
        regime = df['Regime'].iloc[-1] if 'Regime' in df else "NEUTRAL"
        boost = {"BULL": 1.15, "NEUTRAL": 1.0, "BEAR": 0.85}.get(regime, 1.0)
        mt_signal *= boost

        # ── LONG TERME ──
        ticker = [k for k, v in TICKER_TO_PAIR.items() if v == pair]
        ticker = ticker[0] if ticker else pair + "=X"

        # Momentum 12 mois
        if ticker in prices.columns:
            px = prices[ticker].dropna()
            mom_12m = float(np.log(px.iloc[-1] / px.iloc[-252])) if len(px) > 252 else 0
        else:
            mom_12m = 0

        # Carry
        carry_now = 0
        if ticker in carry.columns and len(carry) > 0:
            carry_now = float(carry[ticker].iloc[-1]) * TRADING_DAYS

        lt_signal = 0.6 * np.tanh(mom_12m) + 0.4 * np.tanh(carry_now * 10)

        # ── COMBINÉ ──
        combined = (hw['short'] * ct_signal +
                    hw['medium'] * mt_signal +
                    hw['long'] * lt_signal)

        # Direction et confiance
        direction = "LONG" if combined > 0.05 else ("SHORT" if combined < -0.05 else "FLAT")
        confidence = min(abs(combined) * 100, 100)

        # Sizing (% du capital à allouer)
        sizing = np.clip(abs(combined) * USER.max_weight * 100, 0, USER.max_weight * 100)

        signals.append({
            'Pair': pair,
            'CT_Signal': f"{ct_signal:+.3f}",
            'MT_Signal': f"{mt_signal:+.3f}",
            'LT_Signal': f"{lt_signal:+.3f}",
            'Combined': f"{combined:+.3f}",
            'Direction': direction,
            'Confiance': f"{confidence:.0f}%",
            'Sizing': f"{sizing:.1f}%",
            'Hurst': f"{H:.3f}",
            'Regime': regime,
        })

    df_signals = pd.DataFrame(signals)

    # Affichage
    rows = []
    for _, s in df_signals.iterrows():
        rows.append([
            s['Pair'], s['Direction'], s['Confiance'], s['Sizing'],
            s['CT_Signal'], s['MT_Signal'], s['LT_Signal'], s['Regime']
        ])

    tbl(["Pair", "Signal", "Conf.", "Taille", "CT", "MT", "LT", "Régime"], rows)

    # Résumé
    n_long  = sum(1 for _, s in df_signals.iterrows() if s['Direction'] == 'LONG')
    n_short = sum(1 for _, s in df_signals.iterrows() if s['Direction'] == 'SHORT')
    n_flat  = sum(1 for _, s in df_signals.iterrows() if s['Direction'] == 'FLAT')
    print(f"\n  Résumé : {n_long} LONG | {n_short} SHORT | {n_flat} FLAT")

    # Top 5 convictions
    df_signals['abs_combined'] = df_signals['Combined'].apply(lambda x: abs(float(x)))
    top5 = df_signals.nlargest(5, 'abs_combined')
    print("\n  Top 5 convictions :")
    for _, s in top5.iterrows():
        print(f"    {s['Direction']:5} {s['Pair']:6}  confiance={s['Confiance']}  "
              f"taille={s['Sizing']}  (H={s['Hurst']}, régime={s['Regime']})")

    return df_signals