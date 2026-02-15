from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────────────

class Horizon(str, Enum):
    ONE_MONTH = "1M"
    THREE_MONTHS = "3M"
    SIX_MONTHS = "6M"
    ONE_YEAR = "1Y"
    CUSTOM = "custom"

class OptModel(str, Enum):
    MARKOWITZ = "markowitz"
    ENTROPY_POOLING = "entropy_pooling"
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class FxUniverse(str, Enum):
    MAJORS = "majors"
    MINORS = "minors"
    EXOTICS = "exotics"
    MIX = "mix"

class AnalysisHorizon(str, Enum):
    ONE_DAY = "1"
    FIVE_DAYS = "5"
    THIRTY_DAYS = "30"

class Mode(str, Enum):
    REAL = "real"
    BACKTEST = "backtest"

class Signal(str, Enum):
    STRONG_SELL = "strong_sell"
    SELL = "sell"
    NEUTRAL = "neutral"
    BUY = "buy"
    STRONG_BUY = "strong_buy"


# ── Wizard / Profile ──────────────────────────────────────────────────────────

class ProfileRequest(BaseModel):
    initial_capital: float = Field(..., gt=0)
    horizon: Horizon
    optimization_model: OptModel
    target_return_pct: float = Field(..., ge=0, le=100)
    risk_aversion_level: RiskLevel
    fx_universe: FxUniverse

class AllocationRow(BaseModel):
    pair: str
    weight: float
    notional: float
    market_value: float
    pnl_today: float
    pnl_total: float

class KpiSummary(BaseModel):
    total_value: float
    daily_pnl: float
    cumulative_pnl: float
    net_exposure_usd: float
    concentration_hhi: float

class ProfileResponse(BaseModel):
    session_id: str
    allocation: List[AllocationRow]
    kpi: KpiSummary


# ── Portfolio ─────────────────────────────────────────────────────────────────

class NavPoint(BaseModel):
    date: str
    nav: float
    peak: float

class PortfolioSummary(BaseModel):
    allocation: List[AllocationRow]
    kpi: KpiSummary
    nav_series: List[NavPoint]


# ── Forecasts ─────────────────────────────────────────────────────────────────

class ForecastPoint(BaseModel):
    date: str
    actual: Optional[float] = None
    predicted: Optional[float] = None
    upper: Optional[float] = None
    lower: Optional[float] = None

class ForecastMetrics(BaseModel):
    rmse: float
    mae: float
    r2: float
    directional_accuracy: float

class ForecastResponse(BaseModel):
    series: List[ForecastPoint]
    metrics: ForecastMetrics
    signal: Signal


# ── Risk ──────────────────────────────────────────────────────────────────────

class RiskCards(BaseModel):
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    annual_vol: float
    max_drawdown: float

class HistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int

class VolPoint(BaseModel):
    date: str
    vol: float

class RiskResponse(BaseModel):
    cards: RiskCards
    histogram: List[HistogramBin]
    vol_series: List[VolPoint]


# ── Scenarios ─────────────────────────────────────────────────────────────────

class ScenarioRequest(BaseModel):
    session_id: str
    scenario_type: str = "custom"        # "custom" | predefined name
    pair: Optional[str] = None
    shock_pct: Optional[float] = None

class ScenarioRow(BaseModel):
    pair: str
    weight_before: float
    pnl_before: float
    pnl_after: float
    contribution: float

class ScenarioResponse(BaseModel):
    rows: List[ScenarioRow]
    total_pnl_before: float
    total_pnl_after: float


# ── Backtest ──────────────────────────────────────────────────────────────────

class BacktestCards(BaseModel):
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float

class BacktestResponse(BaseModel):
    nav_series: List[NavPoint]
    return_histogram: List[HistogramBin]
    cards: BacktestCards


# ── Correlations ──────────────────────────────────────────────────────────────

class CorrelationMatrix(BaseModel):
    pairs: List[str]
    matrix: List[List[float]]

class ScatterPoint(BaseModel):
    pair: str
    ann_return: float
    ann_vol: float

class CorrelationResponse(BaseModel):
    matrix: CorrelationMatrix
    scatter: List[ScatterPoint]


# ── Rebalance / Recompute ─────────────────────────────────────────────────────

class SessionAction(BaseModel):
    session_id: str

class TopBarParams(BaseModel):
    session_id: str
    selected_pair: Optional[str] = None
    analysis_horizon: AnalysisHorizon = AnalysisHorizon.ONE_DAY
    mode: Mode = Mode.REAL
    risk_level: RiskLevel = RiskLevel.MEDIUM
