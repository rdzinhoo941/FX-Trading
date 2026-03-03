"""
Visualisations — equity curves, comparaison, signaux
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List
from config import OUTPUT_DIR, C


def plot_equity_curves(results: List, filename="fig1_equity.png"):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor=C['bg'],
                             gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle("Performance par horizon (OOS walk-forward)",
                 color=C['primary'], fontsize=14, fontweight='bold')
    colors = [C['dim'], C['accent'], C['warning'], C['primary'], C['secondary']]
    for i, res in enumerate(results):
        nav = (1 + res.daily_returns).cumprod()
        col = colors[i % len(colors)]
        lw = 2.0 if i == len(results) - 1 else 1.0
        axes[0].plot(nav.index, nav, color=col, lw=lw, label=res.name)
    axes[0].axhline(1, color=C['dim'], ls='--', lw=0.5)
    axes[0].set_ylabel("NAV (base 1)")
    axes[0].legend(fontsize=8, loc='upper left')
    axes[0].grid(True, alpha=0.2)
    best = results[-1]
    nav = (1 + best.daily_returns).cumprod()
    dd = nav / nav.cummax() - 1
    axes[1].fill_between(dd.index, dd, alpha=0.4, color=C['danger'])
    axes[1].set_ylabel(f"Drawdown ({best.name})")
    axes[1].grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {filename}")


def plot_comparison(results: List, filename="fig2_comparison.png"):
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), facecolor=C['bg'])
    fig.suptitle("Comparaison des horizons", color=C['primary'], fontsize=14)
    names = [r.name.split('. ')[1] if '. ' in r.name else r.name for r in results]
    x = np.arange(len(names))
    colors = [C['dim'], C['accent'], C['warning'], C['primary'], C['secondary']]
    configs = [
        (axes[0], [r.sharpe for r in results], "Sharpe"),
        (axes[1], [r.cagr * 100 for r in results], "CAGR (%)"),
        (axes[2], [r.max_dd * 100 for r in results], "Max DD (%)"),
        (axes[3], [r.sortino for r in results], "Sortino"),
    ]
    for ax, metric, label in configs:
        bars = ax.bar(x, metric, color=[colors[i % len(colors)] for i in range(len(results))])
        ax.set_title(label, color=C['text'])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, fontsize=7, ha='right')
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {filename}")


def plot_signals(data_dict: dict, filename="fig3_signals.png"):
    pair = "EURUSD" if "EURUSD" in data_dict else list(data_dict.keys())[0]
    df = data_dict[pair].tail(500)
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), facecolor=C['bg'])
    fig.suptitle(f"Signaux — {pair}", color=C['primary'], fontsize=14)
    axes[0].plot(df.index, df['Close'], color=C['dim'], alpha=0.5, lw=0.6, label='Brut')
    if 'Close_Clean' in df:
        axes[0].plot(df.index, df['Close_Clean'], color=C['primary'], lw=1.5, label='SureShrink')
    axes[0].set_title("Wavelet", color=C['text']); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.2)
    axes[1].plot(df.index, df['Close'], color=C['dim'], alpha=0.5, lw=0.6)
    if 'Kalman_Smoothed' in df:
        axes[1].plot(df.index, df['Kalman_Smoothed'], color='#ff69b4', lw=1.5, label='Kalman')
    axes[1].set_title("Kalman", color=C['text']); axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.2)
    axes[2].plot(df.index, df['Close'], color=C['text'], lw=0.8)
    if 'Regime' in df:
        for reg, col in [("BULL", C['bull']), ("BEAR", C['bear']), ("NEUTRAL", C['neutral'])]:
            mask = df['Regime'] == reg
            if mask.any():
                axes[2].fill_between(df.index, df['Close'].min(), df['Close'].max(), where=mask, alpha=0.15, color=col, label=reg)
    axes[2].set_title("Regimes", color=C['text']); axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.2)
    if 'Hurst' in df:
        h = df['Hurst'].dropna()
        axes[3].plot(h.index, h, color=C['purple'], lw=1)
        axes[3].axhline(0.5, color=C['danger'], ls='--', lw=0.8)
        axes[3].set_ylim(0.2, 0.8)
    axes[3].set_title("Hurst (Garcin)", color=C['text']); axes[3].grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {filename}")