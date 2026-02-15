"""
Backtest service – historical strategy simulation.
>>> WHERE TO PLUG REAL MODELS: replace generate_backtest() with your
    walk-forward backtester.  Feed allocation weights + price history.
"""

import numpy as np
from datetime import datetime, timedelta
from app.schemas import (
    NavPoint, HistogramBin, BacktestCards, BacktestResponse,
)


def generate_backtest(initial_capital: float = 100_000, seed: int = 42) -> BacktestResponse:
    rng = np.random.default_rng(seed)
    days = 252
    today = datetime(2025, 6, 15)

    # target ending NAV around 110-115 indexed to 100
    target_total = rng.uniform(0.10, 0.15)
    daily_drift = target_total / days
    daily_vol = 0.005

    nav = 100.0
    peak = 100.0
    nav_series = []
    daily_returns = []

    for i in range(days):
        ret = daily_drift + rng.normal(0, daily_vol)
        daily_returns.append(ret)
        nav *= 1 + ret
        peak = max(peak, nav)
        d = (today - timedelta(days=days - i)).strftime("%Y-%m-%d")
        nav_series.append(NavPoint(date=d, nav=round(nav, 4), peak=round(peak, 4)))

    rets = np.array(daily_returns)
    ann_ret = float(np.mean(rets) * 252)
    ann_vol = float(np.std(rets) * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    neg = rets[rets < 0]
    downside_vol = float(np.std(neg) * np.sqrt(252)) if len(neg) > 0 else 1e-6
    sortino = ann_ret / downside_vol

    # max drawdown
    navs = np.array([p.nav for p in nav_series])
    peaks = np.maximum.accumulate(navs)
    dd = (navs - peaks) / peaks
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    cards = BacktestCards(
        ann_return=round(ann_ret * 100, 2),
        ann_vol=round(ann_vol * 100, 2),
        sharpe=round(sharpe, 2),
        sortino=round(sortino, 2),
        calmar=round(calmar, 2),
        max_drawdown=round(max_dd * 100, 2),
    )

    # histogram
    counts, edges = np.histogram(rets * 100, bins=30)
    histogram = [
        HistogramBin(bin_start=round(float(edges[i]), 4),
                     bin_end=round(float(edges[i + 1]), 4),
                     count=int(counts[i]))
        for i in range(len(counts))
    ]

    return BacktestResponse(nav_series=nav_series, return_histogram=histogram, cards=cards)
