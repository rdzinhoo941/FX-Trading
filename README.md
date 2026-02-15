# FX Portfolio Optimizer

A production-style FX portfolio optimization dashboard with a **FastAPI** backend and **Next.js** (React + TypeScript) frontend.

## Architecture

```
fx-optimizer/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI routes
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── services/
│   │       ├── data_service.py          # FX universe definitions
│   │       ├── optimization_service.py  # Portfolio construction
│   │       ├── forecast_service.py      # NAV forecasting
│   │       ├── risk_service.py          # VaR, ES, volatility
│   │       ├── backtest_service.py      # Historical simulation
│   │       ├── scenario_service.py      # Stress testing
│   │       └── correlation_service.py   # Cross-pair analytics
│   └── requirements.txt
├── frontend/
│   ├── pages/
│   │   ├── index.tsx       # Wizard (investment profile)
│   │   └── dashboard.tsx   # Tabbed dashboard
│   ├── components/
│   │   ├── KpiCard.tsx
│   │   ├── PositionsTable.tsx
│   │   ├── ChartNav.tsx
│   │   ├── ChartDonut.tsx
│   │   ├── ChartForecast.tsx
│   │   ├── ChartHistogram.tsx
│   │   ├── ChartHeatmap.tsx
│   │   ├── ChartScatter.tsx
│   │   ├── ChartVol.tsx
│   │   ├── TopBar.tsx
│   │   └── Tabs.tsx
│   ├── lib/
│   │   ├── api.ts          # Fetch helpers
│   │   └── types.ts        # TypeScript interfaces
│   ├── styles/globals.css
│   ├── package.json
│   ├── next.config.js      # Proxy /api → backend:8000
│   ├── tailwind.config.js
│   └── tsconfig.json
└── README.md
```

## Quick Start

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Backend runs at **http://localhost:8000**. Swagger docs at **/docs**.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at **http://localhost:3000**. API calls are proxied to the backend via `next.config.js`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/profile/generate_portfolio` | Wizard: generate initial portfolio |
| `GET`  | `/api/portfolio/summary?session_id=...` | Portfolio positions + NAV |
| `POST` | `/api/portfolio/rebalance` | Rebalance portfolio |
| `GET`  | `/api/forecasts?session_id=...` | NAV forecast + metrics |
| `POST` | `/api/forecasts/recompute` | Recompute forecasts |
| `GET`  | `/api/risk/metrics?session_id=...` | Risk analytics |
| `POST` | `/api/scenarios/apply` | Stress test |
| `GET`  | `/api/backtest/results?session_id=...` | Backtest results |
| `GET`  | `/api/correlations/matrix?session_id=...` | Correlation data |

## Example Payloads

### POST /api/profile/generate_portfolio

```json
{
  "initial_capital": 100000,
  "horizon": "3M",
  "optimization_model": "markowitz",
  "target_return_pct": 8,
  "risk_aversion_level": "medium",
  "fx_universe": "majors"
}
```

**Response:**
```json
{
  "session_id": "abc-123-...",
  "allocation": [
    { "pair": "EUR/USD", "weight": 22.5, "notional": 22500, "market_value": 22780, "pnl_today": 12.5, "pnl_total": 280 },
    ...
  ],
  "kpi": {
    "total_value": 101200,
    "daily_pnl": 85.5,
    "cumulative_pnl": 1200,
    "net_exposure_usd": 86020,
    "concentration_hhi": 0.1823
  }
}
```

### POST /api/scenarios/apply

```json
{
  "session_id": "abc-123-...",
  "scenario_type": "custom",
  "pair": "EUR/USD",
  "shock_pct": -5.0
}
```

## Where to Plug Real Models

Each service file in `backend/app/services/` has a docstring marking **WHERE TO PLUG REAL MODELS**:

- **`optimization_service.py`** → Replace `generate_allocation()` with Markowitz, Black-Litterman, entropy pooling, or risk parity optimizer
- **`forecast_service.py`** → Replace `generate_forecast()` with LSTM, Prophet, GARCH, or ensemble model
- **`risk_service.py`** → Replace `generate_risk_metrics()` with parametric/historical VaR, Monte-Carlo ES
- **`backtest_service.py`** → Replace `generate_backtest()` with walk-forward backtester
- **`scenario_service.py`** → Replace `apply_scenario()` with factor-based stress test engine
- **`correlation_service.py`** → Replace `generate_correlations()` with DCC-GARCH or shrinkage estimator
- **`data_service.py`** → Replace pair lookups with live market data feeds

## Tech Stack

- **Backend:** Python 3.11+, FastAPI, Pydantic, NumPy
- **Frontend:** Next.js 14, React 18, TypeScript, Tailwind CSS, Recharts
- **Communication:** JSON REST API
- **State:** In-memory session store (backend), localStorage session_id (frontend)
