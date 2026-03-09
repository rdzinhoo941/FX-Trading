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


def run_fx_framework_allocation(
    pairs: List[str],
    initial_capital: float,
    risk_aversion: str,
    rebalance_mode: str = "weekly",
) -> tuple[List[AllocationRow], KpiSummary]:
    """
    Appelle fx_framework.py et reconstruit les derniers poids de la stratégie combinée.
    rebalance_mode: 'weekly' ou 'monthly'
    """
    repo_root = Path(__file__).resolve().parents[3]
    fx_path = repo_root / "fx_framework.py"

    if not fx_path.exists():
        raise FileNotFoundError(
            f"fx_framework.py introuvable à l'emplacement attendu: {fx_path}"
        )

    fx = _load_module_from_path("fx_framework_runtime", fx_path)

    # Profil utilisateur
    fx.USER.set_profile(_risk_to_profile(risk_aversion))
    fx.USER.selected_pairs = pairs

    # Données : on force le dossier racine comme DATA_DIR
    fx.DATA_DIR = str(repo_root)
    fx.FILE_PRICES = str(repo_root / "data_forex_prices.csv")
    fx.FILE_RATES = str(repo_root / "data_fred_rates.csv")

    prices, rates = fx.load_data()
    tickers = fx.get_fx_universe(prices, rates)
    selected_pairs = [fx.ticker_to_pair(t) for t in tickers]

    if not selected_pairs:
        raise ValueError("Aucune paire valide trouvée dans fx_framework pour la sélection demandée.")

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
    data_dict = fx.apply_hurst(data_dict)
    data_dict = fx.apply_regimes(data_dict)
    data_dict = fx.apply_garch(data_dict)

    ridge_res = fx.train_ridge_ct(data_dict, selected_pairs)
    lgbm_res = fx.train_lgbm_ct(data_dict, selected_pairs)
    gb_res = fx.train_gb_mt(data_dict, selected_pairs)
    trans_res = fx.train_transformer_mt(data_dict, selected_pairs)

    # Recalcule les derniers poids de la stratégie combinée
    hw = fx.USER.horizon_weights
    assets = ret.columns.tolist()
    n = len(assets)
    dates = ret.loc[fx.BACKTEST_START:].index

    if len(dates) == 0:
        raise ValueError("Pas de dates OOS disponibles dans fx_framework.")

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

            df = data_dict[pair]
            if "Regime" in df.columns:
                reg = df["Regime"].loc[:dt].dropna()
                if len(reg) > 0:
                    boost = {"LOW-VOL": 1.1, "HIGH-VOL": 0.80}.get(reg.iloc[-1], 1.0)
                    ct_sig *= boost
                    mt_sig *= boost

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
        pair_ret = float(latest_rets.get(ticker, 0.0)) if ticker in latest_rets.index else 0.0

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
    Placeholder temporaire pour le modèle trimestriel B-L.
    À remplacer dès que tu m’envoies le script/fonction exacte.
    """
    n = len(pairs)
    if n == 0:
        raise ValueError("Aucune paire fournie.")

    w = np.ones(n) / n
    rows: List[AllocationRow] = []

    for pair, weight in zip(pairs, w):
        notional = weight * initial_capital
        rows.append(
            AllocationRow(
                pair=pair,
                weight=round(weight * 100, 2),
                notional=round(notional, 2),
                market_value=round(notional, 2),
                pnl_today=0.0,
                pnl_total=0.0,
            )
        )

    total_mv = sum(r.market_value for r in rows)
    hhi = sum((r.weight / 100) ** 2 for r in rows)

    kpi = KpiSummary(
        total_value=round(total_mv, 2),
        daily_pnl=0.0,
        cumulative_pnl=0.0,
        net_exposure_usd=round(total_mv, 2),
        concentration_hhi=round(hhi, 4),
    )

    return rows, kpi
