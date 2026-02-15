"""
FastAPI backend for FX Portfolio Optimizer.
Run with: uvicorn app.main:app --reload --port 8000
"""

import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    ProfileRequest, ProfileResponse,
    PortfolioSummary, NavPoint,
    ForecastResponse,
    RiskResponse,
    ScenarioRequest, ScenarioResponse,
    BacktestResponse,
    CorrelationResponse,
    SessionAction, TopBarParams,
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

# ── In-memory session store ───────────────────────────────────────────────────
sessions: dict[str, dict] = {}


def _get_session(session_id: str) -> dict:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/profile/generate_portfolio", response_model=ProfileResponse)
def generate_portfolio(req: ProfileRequest):
    sid = str(uuid.uuid4())
    pairs = get_pairs(req.fx_universe.value)
    allocation, kpi = generate_allocation(
        pairs=pairs,
        initial_capital=req.initial_capital,
        model=req.optimization_model.value,
        target_return_pct=req.target_return_pct,
        risk_aversion=req.risk_aversion_level.value,
    )
    sessions[sid] = {
        "profile": req.model_dump(),
        "pairs": pairs,
        "allocation": allocation,
        "kpi": kpi,
        "initial_capital": req.initial_capital,
    }
    return ProfileResponse(session_id=sid, allocation=allocation, kpi=kpi)


@app.get("/api/portfolio/summary", response_model=PortfolioSummary)
def portfolio_summary(session_id: str):
    s = _get_session(session_id)
    rng = np.random.default_rng(42)
    today = datetime(2025, 6, 15)
    nav = 100.0
    peak = 100.0
    nav_series = []
    for i in range(90):
        nav *= 1 + rng.normal(0.0003, 0.005)
        peak = max(peak, nav)
        d = (today - timedelta(days=90 - i)).strftime("%Y-%m-%d")
        nav_series.append(NavPoint(date=d, nav=round(nav, 4), peak=round(peak, 4)))

    return PortfolioSummary(
        allocation=s["allocation"],
        kpi=s["kpi"],
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


@app.post("/api/forecasts/recompute", response_model=ForecastResponse)
def recompute_forecasts(body: SessionAction):
    s = _get_session(body.session_id)
    return generate_forecast(nav_last=s["kpi"].total_value, seed=np.random.randint(0, 10000))


@app.get("/api/forecasts", response_model=ForecastResponse)
def get_forecasts(session_id: str):
    s = _get_session(session_id)
    return generate_forecast(nav_last=s["kpi"].total_value)


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
