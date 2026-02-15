"""
Scenario service – stress-test / what-if analysis.
>>> WHERE TO PLUG REAL MODELS: replace apply_scenario() with factor-based
    stress tests, or Monte-Carlo shocking.
"""

import numpy as np
from app.schemas import AllocationRow, ScenarioRow, ScenarioResponse

PREDEFINED = {
    "usd_crash": {"pair": "EUR/USD", "shock_pct": 5.0},
    "jpy_spike": {"pair": "USD/JPY", "shock_pct": -4.0},
    "risk_off":  {"pair": "GBP/USD", "shock_pct": -3.0},
    "em_selloff": {"pair": "USD/TRY", "shock_pct": 8.0},
}


def apply_scenario(
    allocation: list[AllocationRow],
    scenario_type: str,
    pair: str | None = None,
    shock_pct: float | None = None,
    seed: int = 42,
) -> ScenarioResponse:
    rng = np.random.default_rng(seed)

    if scenario_type != "custom" and scenario_type in PREDEFINED:
        pair = PREDEFINED[scenario_type]["pair"]
        shock_pct = PREDEFINED[scenario_type]["shock_pct"]

    if shock_pct is None:
        shock_pct = -3.0

    rows = []
    total_before = 0.0
    total_after = 0.0

    for pos in allocation:
        pnl_before = pos.pnl_total
        if pair and pos.pair == pair:
            delta = pos.market_value * (shock_pct / 100)
        else:
            # second-order spill
            delta = pos.market_value * rng.normal(0, 0.005)
        pnl_after = pnl_before + delta
        contribution = delta
        total_before += pnl_before
        total_after += pnl_after

        rows.append(ScenarioRow(
            pair=pos.pair,
            weight_before=pos.weight,
            pnl_before=round(pnl_before, 2),
            pnl_after=round(pnl_after, 2),
            contribution=round(contribution, 2),
        ))

    return ScenarioResponse(
        rows=rows,
        total_pnl_before=round(total_before, 2),
        total_pnl_after=round(total_after, 2),
    )
