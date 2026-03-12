"""
FastAPI backend for FX Portfolio Optimizer.
Run with: uvicorn app.main:app --reload --port 8000
"""

import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from typing import Dict

SESSIONS: Dict[str, dict] = {}

def _get_session(session_id: str) -> dict:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    return SESSIONS[session_id]

from app.schemas import (
    ProfileRequest, ProfileResponse,
    PortfolioSummary, NavPoint,
    ForecastResponse,
    RiskResponse,
    ScenarioRequest, ScenarioResponse,
    BacktestResponse,
    CorrelationResponse,
    SessionAction, TopBarParams,
    AllocationRow, KpiSummary,
)
from app.services.data_service import get_pairs
from app.services.optimization_service import generate_allocation
from app.services.forecast_service import generate_forecast
from app.services.risk_service import generate_risk_metrics
from app.services.backtest_service import generate_backtest
from app.services.scenario_service import apply_scenario
from app.services.correlation_service import generate_correlations

import numpy as np
from datetime import datetime, timedelta

app = FastAPI(title="FX Portfolio Optimizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/profile/generate_portfolio")
def generate_portfolio(req: ProfileRequest):
    try:
        print("DEBUG 1 - request received")

        fx_universe = req.fx_universe.value if hasattr(req.fx_universe, "value") else str(req.fx_universe)

        if fx_universe == "majors":
            pairs = [
                "EUR/USD",
                "USD/JPY",
                "GBP/USD",
                "USD/CHF",
                "AUD/USD",
                "USD/CAD",
                "NZD/USD",
            ]
        elif fx_universe == "minors":
            pairs = [
                "EUR/GBP",
                "EUR/JPY",
                "AUD/JPY",
                "NZD/JPY",
            ]
        elif fx_universe == "exotics":
            pairs = [
                "USD/TRY",
                "USD/ZAR",
                "USD/MXN",
                "USD/SGD",
                "USD/NOK",
                "USD/SEK",
            ]
        else:  # mix
            pairs = [
                "EUR/USD",
                "USD/JPY",
                "GBP/USD",
                "AUD/USD",
                "EUR/JPY",
                "AUD/JPY",
                "USD/ZAR",
                "USD/MXN",
            ]

        print("DEBUG fx_universe:", fx_universe)
        print("DEBUG selected pairs:", pairs)
        print("DEBUG 2 - pairs:", pairs)

        allocation, kpi = generate_allocation(
            pairs=pairs,
            initial_capital=req.initial_capital,
            model=req.optimization_model,
            target_return_pct=req.target_return_pct,
            risk_aversion=req.risk_aversion_level,
        )
        print("DEBUG 3 - allocation generated")
        print("DEBUG 3bis - allocation len:", len(allocation))
        print("DEBUG 3ter - kpi:", kpi)

        import uuid
        session_id = str(uuid.uuid4())

        SESSIONS[session_id] = {
    "profile": req.model_dump(),
    "pairs": pairs,
    "initial_capital": req.initial_capital,
    "allocation": [row.model_dump() for row in allocation],
    "kpi": kpi.model_dump(),
        }	

        print("DEBUG 4 - before response")

        response = {
            "session_id": session_id,
            "allocation": [row.model_dump() for row in allocation],
            "kpi": kpi.model_dump(),
        }

        return response

    except Exception as e:
        import traceback
        print("DEBUG ERROR:", repr(e))
        traceback.print_exc()
        raise

from pathlib import Path
import pandas as pd


def _build_dynamic_nav_series(allocation, initial_capital: float, lookback_days: int = 90):
    """
    Build a dynamic NAV series from current portfolio weights applied
    to recent historical daily FX returns.

    This is not a full backtest with rolling rebalancing.
    It is a dynamic historical NAV based on the portfolio's current weights.
    """
    repo_root = Path(__file__).resolve().parents[2]

    prices_candidates = [
        repo_root / "data" / "data_forex_prices.csv",
        repo_root / "data_forex_prices.csv",
    ]
    prices_path = next((p for p in prices_candidates if p.exists()), None)

    if prices_path is None:
        raise FileNotFoundError("Could not find data_forex_prices.csv")

    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True).sort_index()

    weight_map = {}
    for row in allocation:
        pair = row.pair if hasattr(row, "pair") else row["pair"]
        weight_pct = row.weight if hasattr(row, "weight") else row["weight"]
        ticker = f"{pair}=X"
        if ticker in prices.columns:
            weight_map[ticker] = float(weight_pct) / 100.0

    if not weight_map:
        return [
            NavPoint(
                date="2025-01-01",
                nav=round(initial_capital, 4),
                peak=round(initial_capital, 4),
            )
        ]

    selected_prices = prices[list(weight_map.keys())].dropna(how="all").ffill().dropna()

    if len(selected_prices) < 2:
        return [
            NavPoint(
                date="2025-01-01",
                nav=round(initial_capital, 4),
                peak=round(initial_capital, 4),
            )
        ]

    selected_prices = selected_prices.tail(lookback_days)
    returns = selected_prices.pct_change().dropna()

    weights = pd.Series(weight_map)
    weights = weights.reindex(returns.columns).fillna(0.0)

    portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)

    nav = initial_capital * (1 + portfolio_returns).cumprod()
    peak = nav.cummax()

    nav_series = [
        NavPoint(
            date=dt.strftime("%Y-%m-%d"),
            nav=round(float(nv), 4),
            peak=round(float(pk), 4),
        )
        for dt, nv, pk in zip(nav.index, nav.values, peak.values)
    ]

    return nav_series


def _horizon_to_lookback_days(horizon) -> int:
    """
    Map profile horizon to approximate trading days.
    """
    h = horizon.value if hasattr(horizon, "value") else str(horizon)

    if h == "1M":
        return 21
    elif h == "3M":
        return 63
    elif h == "6M":
        return 126
    elif h == "1Y":
        return 252
    else:
        return 90

@app.get("/api/portfolio/summary", response_model=PortfolioSummary)
def portfolio_summary(session_id: str):
    s = _get_session(session_id)

    allocation = [
        row if isinstance(row, AllocationRow) else AllocationRow(**row)
        for row in s["allocation"]
    ]
    kpi = s["kpi"] if isinstance(s["kpi"], KpiSummary) else KpiSummary(**s["kpi"])

    initial_capital = s.get("initial_capital", 100000.0)

    profile = s.get("profile", {})
    horizon = profile.get("horizon", "3M") if isinstance(profile, dict) else "3M"
    lookback_days = _horizon_to_lookback_days(horizon)

    print("DEBUG profile in session:", profile)
    print("DEBUG horizon used:", horizon)
    print("DEBUG lookback_days used:", lookback_days)

    nav_series = _build_dynamic_nav_series(
        allocation,
        initial_capital=initial_capital,
        lookback_days=lookback_days,
    )

    return PortfolioSummary(
        allocation=allocation,
        kpi=kpi,
        nav_series=nav_series,
    )

@app.post("/api/portfolio/rebalance", response_model=ProfileResponse)
def rebalance(body: SessionAction):
    s = _get_session(body.session_id)
    p = s["profile"]
    allocation, kpi = generate_allocation(
        pairs=s["pairs"],
        initial_capital=s["initial_capital"],
        model=p["optimization_model"],
        target_return_pct=p["target_return_pct"],
        risk_aversion=p["risk_aversion_level"],
        seed=np.random.randint(0, 10000),
    )
    s["allocation"] = allocation
    s["kpi"] = kpi
    return ProfileResponse(session_id=body.session_id, allocation=allocation, kpi=kpi)


@app.post("/api/forecasts/recompute")
def recompute_forecasts(req: SessionAction):
    s = _get_session(req.session_id)

    nav_last = s["kpi"].total_value if isinstance(s["kpi"], KpiSummary) else s["kpi"]["total_value"]

    return generate_forecast(
        nav_last=nav_last,
        seed=np.random.randint(0, 10000),
    )

@app.get("/api/forecasts", response_model=ForecastResponse)
def get_forecasts(session_id: str):
    s = _get_session(session_id)
    nav_last = s["kpi"].total_value if isinstance(s["kpi"], KpiSummary) else s["kpi"]["total_value"]
    return generate_forecast(nav_last=nav_last)


@app.get("/api/risk/metrics", response_model=RiskResponse)
def risk_metrics(session_id: str):
    s = _get_session(session_id)
    return generate_risk_metrics(initial_capital=s["initial_capital"])


@app.post("/api/scenarios/apply", response_model=ScenarioResponse)
def scenarios_apply(req: ScenarioRequest):
    s = _get_session(req.session_id)
    return apply_scenario(
        allocation=s["allocation"],
        scenario_type=req.scenario_type,
        pair=req.pair,
        shock_pct=req.shock_pct,
    )


@app.get("/api/backtest/results", response_model=BacktestResponse)
def backtest_results(session_id: str):
    s = _get_session(session_id)
    return generate_backtest(initial_capital=s["initial_capital"])


@app.get("/api/correlations/matrix", response_model=CorrelationResponse)
def correlations_matrix(session_id: str):
    s = _get_session(session_id)
    return generate_correlations(pairs=s["pairs"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
