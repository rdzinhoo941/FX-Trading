"""
Forecast service – NAV-level forecasting.
>>> WHERE TO PLUG REAL MODELS: replace generate_forecast() with your
    LSTM / Prophet / GARCH model.  Feed it portfolio NAV history.
"""

import numpy as np
from datetime import datetime, timedelta
from app.schemas import ForecastPoint, ForecastMetrics, ForecastResponse, Signal


def generate_forecast(nav_last: float = 101_000, seed: int = 42) -> ForecastResponse:
    rng = np.random.default_rng(seed)
    today = datetime(2025, 6, 15)
    n_hist = 60
    n_pred = 20

    # historical NAV
    hist_rets = rng.normal(0.0004, 0.006, n_hist)
    nav = nav_last * 0.97
    series: list[ForecastPoint] = []
    for i in range(n_hist):
        nav *= 1 + hist_rets[i]
        d = (today - timedelta(days=n_hist - i)).strftime("%Y-%m-%d")
        series.append(ForecastPoint(date=d, actual=round(nav, 2)))

    actual_end = nav

    # predictions (overlaps last 10 hist + 10 future)
    pred_nav = actual_end
    for i in range(n_pred):
        d = (today + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        drift = 0.0003
        pred_nav *= 1 + drift + rng.normal(0, 0.003)
        band = pred_nav * 0.015 * (1 + i * 0.1)
        series.append(ForecastPoint(
            date=d,
            predicted=round(pred_nav, 2),
            upper=round(pred_nav + band, 2),
            lower=round(pred_nav - band, 2),
        ))

    metrics = ForecastMetrics(
        rmse=round(rng.uniform(180, 350), 2),
        mae=round(rng.uniform(120, 250), 2),
        r2=round(rng.uniform(0.82, 0.94), 4),
        directional_accuracy=round(rng.uniform(0.58, 0.72), 4),
    )
    signal = Signal.BUY

    return ForecastResponse(series=series, metrics=metrics, signal=signal)
