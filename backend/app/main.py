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

        pairs = [
            "EUR/USD",
            "USD/JPY",
            "GBP/USD",
            "USD/CHF",
            "AUD/USD",
            "USD/CAD",
            "NZD/USD",
        ]
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

@app.get("/api/portfolio/summary", response_model=PortfolioSummary)
def portfolio_summary(session_id: str):
    s = _get_session(session_id)

    allocation = [
    row if isinstance(row, AllocationRow) else AllocationRow(**row)
    for row in s["allocation"]
    ]
    kpi = s["kpi"] if isinstance(s["kpi"], KpiSummary) else KpiSummary(**s["kpi"])

    nav_series = [
    NavPoint(date="2024-12-28", nav=100000.0, peak=100000.0),
    NavPoint(date="2024-12-29", nav=100250.0, peak=100250.0),
    NavPoint(date="2024-12-30", nav=99850.0, peak=100250.0),
    NavPoint(date="2024-12-31", nav=100400.0, peak=100400.0),
    NavPoint(date="2025-01-01", nav=round(kpi.total_value, 4), peak=max(100400.0, round(kpi.total_value, 4))),
    ]

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
