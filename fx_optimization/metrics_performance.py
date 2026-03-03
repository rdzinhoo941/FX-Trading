"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  MÉTRIQUES DE PERFORMANCE                                                      ║
║                                                                                ║
║  On définit ici toutes les métriques qu'on va utiliser pour comparer les       ║
║  stratégies.                                                                   ║
║                                                                                ║
║  • Sharpe  = rendement excédentaire / volatilité                               ║
║  • Sortino = comme Sharpe mais ne pénalise que la vol baissière                ║
║  • Calmar  = CAGR / max drawdown                                               ║
║  • Max DD  = pire perte peak-to-trough                                         ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from config import TRADING_DAYS
from utils_display import hdr, tbl

@dataclass
class StrategyResult:
    """Conteneur pour les résultats d'une stratégie."""
    name: str
    daily_returns: pd.Series
    weights_history: Optional[pd.DataFrame] = None
    description: str = ""

    # Métriques calculées après coup
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_dd: float = 0.0
    cagr: float = 0.0
    ann_vol: float = 0.0
    win_rate: float = 0.0
    total_return: float = 0.0

    def compute_metrics(self):
        """Calcule toutes les métriques de performance à partir des rendements journaliers."""
        r = self.daily_returns.dropna()
        if len(r) < 10:
            return self

        nav = (1 + r).cumprod()
        n = len(r)

        # CAGR (Compound Annual Growth Rate)
        self.cagr = float(nav.iloc[-1] ** (TRADING_DAYS / n) - 1)
        self.total_return = float(nav.iloc[-1] - 1)

        # Volatilité annualisée
        self.ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))

        # Sharpe (on considère rf ≈ 0 pour simplifier la comparaison)
        self.sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0

        # Sortino (ne pénalise que la vol baissière)
        downside = r[r < 0].std()
        self.sortino = float(r.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else 0

        # Max Drawdown (pire perte peak-to-trough)
        peak = nav.cummax()
        dd = nav / peak - 1
        self.max_dd = float(dd.min())

        # Calmar (CAGR / Max DD)
        self.calmar = float(self.cagr / abs(self.max_dd)) if self.max_dd != 0 else 0

        # Win rate (pourcentage de jours positifs)
        self.win_rate = float((r > 0).mean())

        return self


def print_comparison_table(results: List[StrategyResult]):
    """Affiche le tableau comparatif de toutes les stratégies."""
    hdr("COMPARAISON DES STRATÉGIES")
    headers = ["#", "Stratégie", "Sharpe", "Sortino", "Calmar", "MaxDD", "CAGR", "Vol", "Win%"]
    rows = []
    for i, res in enumerate(results):
        rows.append([
            str(i), res.name,
            f"{res.sharpe:.3f}", f"{res.sortino:.3f}", f"{res.calmar:.3f}",
            f"{res.max_dd:.2%}", f"{res.cagr:.2%}", f"{res.ann_vol:.2%}",
            f"{res.win_rate:.1%}"
        ])
    tbl(headers, rows)


def summary_to_dataframe(results: List[StrategyResult]) -> pd.DataFrame:
    """Convertit les résultats en DataFrame pour sauvegarde CSV."""
    return pd.DataFrame([{
        'Strategy': r.name,
        'Sharpe': r.sharpe,
        'Sortino': r.sortino,
        'CAGR': r.cagr,
        'MaxDD': r.max_dd,
        'Calmar': r.calmar,
        'AnnVol': r.ann_vol,
        'WinRate': r.win_rate,
        'TotalReturn': r.total_return
    } for r in results])