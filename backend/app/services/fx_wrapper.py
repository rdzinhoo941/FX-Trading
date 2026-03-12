from __future__ import annotations

from pathlib import Path
import importlib.util
from typing import List, Dict

import numpy as np

from app.schemas import AllocationRow, KpiSummary


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _risk_to_profile(risk_aversion: str) -> str:
    mapping = {
        "low": "conservateur",
        "medium": "equilibre",
        "high": "agressif",
    }
    return mapping.get((risk_aversion or "").lower(), "equilibre")


def _normalize_weights(raw_weights: Dict[str, float], requested_pairs: List[str]) -> Dict[str, float]:
    filtered = {p: float(raw_weights.get(p, 0.0)) for p in requested_pairs}
    gross = sum(abs(v) for v in filtered.values())
    if gross <= 1e-12:
        n = len(requested_pairs)
        if n == 0:
            return {}
        return {p: 1.0 / n for p in requested_pairs}
    return {p: v / gross for p, v in filtered.items()}


def _normalize_fx_pairs_for_framework(pairs: List[str]) -> List[str]:
    """
    Convertit les paires envoyées par l'API vers le format attendu par fx_framework.
    Exemples:
    - EUR/USD   -> EURUSD
    - EURUSD=X  -> EURUSD
    - eurusd    -> EURUSD
    """
    allowed = {
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF",
        "EURGBP", "EURJPY", "AUDJPY", "NZDJPY",
        "USDMXN", "USDNOK", "USDSEK", "USDSGD", "USDTRY", "USDZAR",
    }

    normalized = []

    for p in pairs:
        if not p:
            continue

        x = str(p).strip().upper()

        # EUR/USD -> EURUSD
        x = x.replace("/", "")

        # EURUSD=X -> EURUSD
        if x.endswith("=X"):
            x = x[:-2]

        # retire espaces éventuels
        x = x.replace(" ", "")

        if x in allowed:
            normalized.append(x)

    return normalized


def run_fx_framework_allocation(
    pairs: List[str],
    initial_capital: float,
    risk_aversion: str,
    rebalance_mode: str = "weekly",
) -> tuple[List[AllocationRow], KpiSummary]:

    repo_root = Path(__file__).resolve().parents[3]
    fx_path = repo_root / "fx_framework.py"

    if not fx_path.exists():
        raise FileNotFoundError(f"fx_framework.py introuvable: {fx_path}")

    fx = _load_module_from_path("fx_framework_runtime", fx_path)

    # Profil utilisateur
    fx.USER.set_profile(_risk_to_profile(risk_aversion))

    normalized_pairs = _normalize_fx_pairs_for_framework(pairs)
    fx.USER.selected_pairs = normalized_pairs if normalized_pairs else None

    # chemins données
    fx.DATA_DIR = str(repo_root)
    fx.FILE_PRICES = str(repo_root / "data_forex_prices.csv")
    fx.FILE_RATES = str(repo_root / "data_fred_rates.csv")

    prices, rates = fx.load_data()

    tickers = fx.get_fx_universe(prices, rates)
    selected_pairs = [fx.ticker_to_pair(t) for t in tickers]

    if not selected_pairs:
        raise ValueError(
            f"Aucune paire valide trouvée dans fx_framework.\n"
            f"Pairs envoyées par API: {pairs}\n"
            f"Pairs normalisées: {normalized_pairs}"
        )

    ret = fx.compute_returns(prices[tickers])

    data_dict = {}
    for t in tickers:
        pair = fx.ticker_to_pair(t)

        df = prices[[t]].copy()
        df.columns = ["Close"]
        df = df.dropna()

        df["Return_1d"] = df["Close"].pct_change()

        data_dict[pair] = df

    data_dict = fx.apply_wavelet(data_dict)
    data_dict = fx.create_features(data_dict)

# Désactivation temporaire de Hurst car cette étape bloque le pipeline
    for pair, df in data_dict.items():
        df["Hurst"] = 0.5
        df["Fractal_Dimension"] = 1.5
        df["Hurst_Prediction"] = 0.0

    ridge_res = fx.train_ridge_ct(data_dict, selected_pairs)
    lgbm_res = fx.train_lgbm_ct(data_dict, selected_pairs)
    gb_res = fx.train_gb_mt(data_dict, selected_pairs)

# On garde un fallback léger à la place du transformer
    trans_res = gb_res

    hw = fx.USER.horizon_weights

    assets = ret.columns.tolist()
    n = len(assets)

    dates = ret.loc[fx.BACKTEST_START:].index

    rebal_every = 5 if rebalance_mode == "weekly" else 21

    w_prev = np.zeros(n)

    for t_idx, dt in enumerate(dates):

        loc = ret.index.get_loc(dt)

        if t_idx % rebal_every != 0 or loc < 252:
            continue

        signals = np.zeros(n)

        for i, asset in enumerate(assets):

            pair = fx.ticker_to_pair(asset)

            if pair not in data_dict:
                continue

            p_ridge = fx._get_signal(ridge_res.get(pair, {}), dt)
            p_lgbm = fx._get_signal(lgbm_res.get(pair, {}), dt)

            ct_sig = (
                fx.USER.ct_model_weights["ridge"] * (p_ridge - 0.5) * 2
                + fx.USER.ct_model_weights["lgbm"] * (p_lgbm - 0.5) * 2
            )

            p_gb = fx._get_signal(gb_res.get(pair, {}), dt)
            p_tr = fx._get_signal(trans_res.get(pair, {}), dt)

            mt_sig = (
                fx.USER.mt_model_weights["gb"] * (p_gb - 0.5) * 2
                + fx.USER.mt_model_weights["transformer"] * (p_tr - 0.5) * 2
            )

            signals[i] = hw["ct"] * ct_sig + hw["mt"] * mt_sig

        w_new = np.tanh(signals * 1.5)

        window = ret.iloc[max(0, loc - 252):loc]

        cov = window.cov().values + 1e-6 * np.eye(n)

        w_new = fx._vol_scale(w_new, cov, fx.USER.target_vol, fx.USER.max_leverage)

        w_new = np.clip(w_new, -fx.USER.max_weight, fx.USER.max_weight)

        w_prev = w_new

    raw_weights = {
        fx.ticker_to_pair(asset): float(weight)
        for asset, weight in zip(assets, w_prev)
    }

    final_weights = _normalize_weights(raw_weights, selected_pairs)

    latest_dt = ret.index[-1]

    latest_rets = ret.loc[latest_dt]

    rows: List[AllocationRow] = []

    for pair in selected_pairs:

        weight = final_weights.get(pair, 0.0)

        notional = weight * initial_capital

        ticker = fx.PAIR_MAP.get(pair)

        pair_ret = float(latest_rets.get(ticker, 0.0))

        market_value = notional * (1 + pair_ret)

        pnl_today = notional * pair_ret

        pnl_total = market_value - notional

        rows.append(
            AllocationRow(
                pair=pair,
                weight=round(weight * 100, 2),
                notional=round(notional, 2),
                market_value=round(market_value, 2),
                pnl_today=round(pnl_today, 2),
                pnl_total=round(pnl_total, 2),
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
        net_exposure_usd=round(sum(abs(r.market_value) for r in rows), 2),
        concentration_hhi=round(hhi, 4),
    )

    return rows, kpi

def run_quarterly_bl_allocation(
    pairs: List[str],
    initial_capital: float,
    risk_aversion: str,
) -> tuple[List[AllocationRow], KpiSummary]:
    """
    Quarterly Black-Litterman allocation compatible with the backend API.

    Logic kept from the research script:
    - FX total returns = price return + carry
    - momentum-based views
    - covariance-based prior
    - BL posterior expected returns
    - constrained optimization
    """

    from pathlib import Path
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize

    repo_root = Path(__file__).resolve().parents[3]

    # Accept either /data or root-level files
    prices_candidates = [
        repo_root / "data" / "data_forex_prices.csv",
        repo_root / "data_forex_prices.csv",
    ]
    rates_candidates = [
        repo_root / "data" / "data_fred_rates.csv",
        repo_root / "data_fred_rates.csv",
    ]

    prices_path = next((p for p in prices_candidates if p.exists()), None)
    rates_path = next((p for p in rates_candidates if p.exists()), None)

    if prices_path is None or rates_path is None:
        raise FileNotFoundError(
            "Could not find data_forex_prices.csv / data_fred_rates.csv "
            "either in /data or at repo root."
        )

    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True).sort_index()
    rates_daily = pd.read_csv(rates_path, index_col=0, parse_dates=True).sort_index()

    normalized_pairs = _normalize_fx_pairs_for_framework(pairs)
    if not normalized_pairs:
        raise ValueError("No valid FX pairs provided for quarterly BL allocation.")

    requested_tickers = [f"{p}=X" for p in normalized_pairs if f"{p}=X" in prices.columns]
    if not requested_tickers:
        raise ValueError("Requested FX pairs are not available in the price dataset.")

    prices = prices[requested_tickers].dropna(how="all")

    # Align indices
    common_idx = prices.index.intersection(rates_daily.index)
    prices = prices.loc[common_idx].ffill().dropna(how="all")
    rates_daily = rates_daily.loc[common_idx].ffill()

    if len(prices) < 260:
        raise ValueError("Not enough historical data for quarterly BL allocation.")

    # -----------------------------
    # Parameters inspired by teammate's BL interface version
    # -----------------------------
    LOOKBACK_MOMENTUM = 252   # 12 months
    LOOKBACK_COV = 252
    BROKER_SWAP_MARKUP = 0.0001
    TAU = 0.05
    MAX_WEIGHT = 0.50
    GROSS_LEVERAGE = 1.50

    # Higher delta = more risk aversion
    delta_map = {
        "low": 4.0,
        "medium": 2.5,
        "high": 1.5,
    }
    delta = delta_map.get(str(risk_aversion).lower(), 2.5)

    def parse_pair_ticker(ticker: str) -> tuple[str, str]:
        x = ticker.replace("=X", "")
        return x[:3], x[3:]

    # -----------------------------
    # Total returns = price return + carry
    # -----------------------------
    returns_total = pd.DataFrame(index=prices.index, columns=requested_tickers, dtype=float)

    for ticker in requested_tickers:
        base, quote = parse_pair_ticker(ticker)

        price_ret = prices[ticker].pct_change()

        carry_ret = 0.0
        if base in rates_daily.columns and quote in rates_daily.columns:
            raw_diff = rates_daily[base] - rates_daily[quote]
            realistic_diff = raw_diff - BROKER_SWAP_MARKUP
            carry_ret = realistic_diff / 252.0

        returns_total[ticker] = price_ret + carry_ret

    returns_total = returns_total.dropna(how="all")
    if len(returns_total) < LOOKBACK_COV + 5:
        raise ValueError("Not enough total return history for quarterly BL.")

    # -----------------------------
    # Quarterly-style momentum views
    # Use 12m momentum but rebalance on the latest available date
    # -----------------------------
    log_total = np.log1p(returns_total.clip(lower=-0.999999))
    momentum_score = log_total.rolling(LOOKBACK_MOMENTUM).sum().shift(1)
    momentum_score = momentum_score.dropna(how="all")

    if momentum_score.empty:
        raise ValueError("Momentum score could not be computed for quarterly BL.")

    last_date = momentum_score.index[-1]
    views_raw = momentum_score.loc[last_date].fillna(0.0)

    # Normalize views cross-sectionally
    if views_raw.std() > 1e-12:
        views_z = (views_raw - views_raw.mean()) / views_raw.std()
    else:
        views_z = views_raw * 0.0

    # Convert to expected return tilts (keep realistic magnitude)
    Q = (views_z.clip(-2, 2) * 0.02).values.reshape(-1, 1)

    # -----------------------------
    # Covariance
    # -----------------------------
    cov_window = returns_total.loc[:last_date].tail(LOOKBACK_COV)
    Sigma = cov_window.cov().values
    n = len(requested_tickers)
    Sigma = Sigma + 1e-6 * np.eye(n)

    # -----------------------------
    # Prior / equilibrium returns
    # -----------------------------
    w_eq = np.ones(n) / n
    pi = delta * Sigma @ w_eq

    # -----------------------------
    # Black-Litterman posterior
    # -----------------------------
    P = np.eye(n)
    omega_diag = np.maximum(np.diag(P @ (TAU * Sigma) @ P.T), 1e-6)
    Omega = np.diag(omega_diag)

    inv_tau_sigma = np.linalg.inv(TAU * Sigma)
    inv_omega = np.linalg.inv(Omega)

    posterior_cov = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
    posterior_mean = posterior_cov @ (
        inv_tau_sigma @ pi.reshape(-1, 1) + P.T @ inv_omega @ Q
    )
    mu_bl = posterior_mean.flatten()

    # -----------------------------
    # Constrained optimization
    # -----------------------------
    def objective(w: np.ndarray) -> float:
        expected_term = mu_bl @ w
        risk_term = 0.5 * delta * (w @ Sigma @ w)
        return -(expected_term - risk_term)

    constraints = [
        {"type": "ineq", "fun": lambda w: GROSS_LEVERAGE - np.sum(np.abs(w))}
    ]
    bounds = [(-MAX_WEIGHT, MAX_WEIGHT) for _ in range(n)]
    x0 = np.ones(n) / n

    res = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-9},
    )

    if res.success:
        raw_w = res.x
    else:
        # fallback to normalized views if optimization fails
        raw_w = views_z.values.astype(float)
        gross = np.sum(np.abs(raw_w))
        if gross <= 1e-12:
            raw_w = np.ones(n) / n
        else:
            raw_w = raw_w / gross
        raw_w = np.clip(raw_w, -MAX_WEIGHT, MAX_WEIGHT)

    gross = np.sum(np.abs(raw_w))
    if gross <= 1e-12:
        raw_w = np.ones(n) / n
    elif gross > GROSS_LEVERAGE:
        raw_w = raw_w * (GROSS_LEVERAGE / gross)

    raw_weights = {
        ticker.replace("=X", ""): float(w)
        for ticker, w in zip(requested_tickers, raw_w)
    }

    final_weights = _normalize_weights(
        raw_weights,
        [t.replace("=X", "") for t in requested_tickers]
    )

    latest_ret = returns_total.loc[last_date]

    rows: List[AllocationRow] = []
    for pair in [t.replace("=X", "") for t in requested_tickers]:
        weight = final_weights.get(pair, 0.0)
        notional = weight * initial_capital

        ticker = f"{pair}=X"
        pair_ret = float(latest_ret.get(ticker, 0.0))

        market_value = notional * (1 + pair_ret)
        pnl_today = notional * pair_ret
        pnl_total = market_value - notional

        rows.append(
            AllocationRow(
                pair=pair,
                weight=round(weight * 100, 2),
                notional=round(notional, 2),
                market_value=round(market_value, 2),
                pnl_today=round(pnl_today, 2),
                pnl_total=round(pnl_total, 2),
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
        net_exposure_usd=round(sum(abs(r.market_value) for r in rows), 2),
        concentration_hhi=round(hhi, 4),
    )

    return rows, kpi