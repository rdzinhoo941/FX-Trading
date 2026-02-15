"""
Risk service – portfolio risk analytics.
>>> WHERE TO PLUG REAL MODELS: replace generate_risk_metrics() with
    historical/parametric VaR, Monte-Carlo ES, GARCH vol, etc.
"""

import numpy as np
from datetime import datetime, timedelta
from app.schemas import (
    RiskCards, HistogramBin, VolPoint, RiskResponse,
)


def generate_risk_metrics(initial_capital: float = 100_000, seed: int = 42) -> RiskResponse:
    rng = np.random.default_rng(seed)

    # cards
    cards = RiskCards(
        var_95=round(-initial_capital * rng.uniform(0.008, 0.015), 2),
        var_99=round(-initial_capital * rng.uniform(0.015, 0.025), 2),
        es_95=round(-initial_capital * rng.uniform(0.012, 0.02), 2),
        es_99=round(-initial_capital * rng.uniform(0.02, 0.035), 2),
        annual_vol=round(rng.uniform(6.5, 11.0), 2),
        max_drawdown=round(-rng.uniform(3.0, 8.0), 2),
    )

    # histogram of daily returns – slightly positive mean
    returns = rng.normal(0.02, 0.8, 500)
    counts, edges = np.histogram(returns, bins=30)
    histogram = [
        HistogramBin(bin_start=round(float(edges[i]), 4),
                     bin_end=round(float(edges[i + 1]), 4),
                     count=int(counts[i]))
        for i in range(len(counts))
    ]

    # realized vol series
    today = datetime(2025, 6, 15)
    vol_series = []
    vol = 8.0
    for i in range(120):
        vol += rng.normal(0, 0.3)
        vol = max(4.0, min(16.0, vol))
        d = (today - timedelta(days=120 - i)).strftime("%Y-%m-%d")
        vol_series.append(VolPoint(date=d, vol=round(vol, 2)))

    return RiskResponse(cards=cards, histogram=histogram, vol_series=vol_series)
