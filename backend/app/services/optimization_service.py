"""
Optimization service – portfolio construction.
>>> WHERE TO PLUG REAL MODELS: replace generate_allocation() with your
    Markowitz / entropy-pooling / risk-parity optimiser.  The function
    signature and return types should stay the same.
"""

import numpy as np
from app.schemas import AllocationRow, KpiSummary


def generate_allocation(
    pairs: list[str],
    initial_capital: float,
    model: str,
    target_return_pct: float,
    risk_aversion: str,
    seed: int = 42,
) -> tuple[list[AllocationRow], KpiSummary]:
    rng = np.random.default_rng(seed)
    n = len(pairs)

    # weights
    if model == "equal_weight":
        w = np.ones(n) / n
    elif model == "risk_parity":
        raw = rng.uniform(0.8, 1.2, n)
        inv_vol = 1.0 / raw
        w = inv_vol / inv_vol.sum()
    else:
        raw = rng.dirichlet(np.ones(n) * 3)
        w = raw

    notionals = w * initial_capital
    # small realistic daily pnl
    daily_rets = rng.normal(0.0003, 0.004, n)
    total_rets = rng.normal(0.012, 0.02, n)

    rows = []
    for i, pair in enumerate(pairs):
        mv = notionals[i] * (1 + total_rets[i])
        rows.append(AllocationRow(
            pair=pair,
            weight=round(float(w[i]) * 100, 2),
            notional=round(float(notionals[i]), 2),
            market_value=round(float(mv), 2),
            pnl_today=round(float(notionals[i] * daily_rets[i]), 2),
            pnl_total=round(float(mv - notionals[i]), 2),
        ))

    total_mv = sum(r.market_value for r in rows)
    daily = sum(r.pnl_today for r in rows)
    cum = sum(r.pnl_total for r in rows)
    hhi = sum((r.weight / 100) ** 2 for r in rows)

    kpi = KpiSummary(
        total_value=round(total_mv, 2),
        daily_pnl=round(daily, 2),
        cumulative_pnl=round(cum, 2),
        net_exposure_usd=round(total_mv * 0.85, 2),
        concentration_hhi=round(hhi, 4),
    )
    return rows, kpi
