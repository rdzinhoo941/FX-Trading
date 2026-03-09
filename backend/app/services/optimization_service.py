"""
Optimization service – portfolio construction.

This file is the central entry point used by the FastAPI backend
to construct a portfolio allocation.

We dispatch to different models depending on the optimization_model
selected in the frontend wizard.
"""

from __future__ import annotations

from typing import List
import numpy as np

from app.schemas import AllocationRow, KpiSummary
from app.services.fx_wrapper import (
    run_fx_framework_allocation,
    run_quarterly_bl_allocation,
)


# -------------------------------------------------------------------
# Fallback model (kept for legacy models not yet connected)
# -------------------------------------------------------------------

def _fallback_mock_allocation(
    pairs: List[str],
    initial_capital: float,
    model: str,
    seed: int = 42,
) -> tuple[List[AllocationRow], KpiSummary]:

    rng = np.random.default_rng(seed)
    n = len(pairs)

    if n == 0:
        raise ValueError("No FX pairs provided.")

    # portfolio weights
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

    # simulated returns
    daily_rets = rng.normal(0.0003, 0.004, n)
    total_rets = rng.normal(0.012, 0.02, n)

    rows: List[AllocationRow] = []

    for i, pair in enumerate(pairs):

        mv = notionals[i] * (1 + total_rets[i])

        rows.append(
            AllocationRow(
                pair=pair,
                weight=round(float(w[i]) * 100, 2),
                notional=round(float(notionals[i]), 2),
                market_value=round(float(mv), 2),
                pnl_today=round(float(notionals[i] * daily_rets[i]), 2),
                pnl_total=round(float(mv - notionals[i]), 2),
            )
        )

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


# -------------------------------------------------------------------
# Main portfolio allocation dispatcher
# -------------------------------------------------------------------

def generate_allocation(
    pairs: List[str],
    initial_capital: float,
    model: str,
    target_return_pct: float,
    risk_aversion: str,
    seed: int = 42,
) -> tuple[List[AllocationRow], KpiSummary]:
    """
    Main portfolio construction entry point.

    This function is called by the FastAPI endpoint and dispatches
    to the correct portfolio construction model.
    """

    # ---------------------------------------------------------------
    # Weekly ML FX model (Thomas)
    # ---------------------------------------------------------------
    if model == "weekly_rebalance":

        return run_fx_framework_allocation(
            pairs=pairs,
            initial_capital=initial_capital,
            risk_aversion=risk_aversion,
            rebalance_mode="weekly",
        )

    # ---------------------------------------------------------------
    # Monthly ML FX model (Thomas)
    # ---------------------------------------------------------------
    if model == "monthly_rebalance":

        return run_fx_framework_allocation(
            pairs=pairs,
            initial_capital=initial_capital,
            risk_aversion=risk_aversion,
            rebalance_mode="monthly",
        )

    # ---------------------------------------------------------------
    # Quarterly Black-Litterman momentum model
    # ---------------------------------------------------------------
    if model == "quarterly_bl":

        return run_quarterly_bl_allocation(
            pairs=pairs,
            initial_capital=initial_capital,
            risk_aversion=risk_aversion,
        )

    # ---------------------------------------------------------------
    # Legacy / fallback models
    # ---------------------------------------------------------------
    return _fallback_mock_allocation(
        pairs=pairs,
        initial_capital=initial_capital,
        model=model,
        seed=seed,
    )
