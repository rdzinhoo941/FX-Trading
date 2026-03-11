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
    Wrapper API pour le script 'B-L momentum for assets and rates.py'.

    On reconstruit les poids BL à la dernière date disponible,
    puis on les convertit au format attendu par l'API.
    """
    repo_root = Path(__file__).resolve().parents[3]
    bl_path = repo_root / "B-L momentum for assets and rates.py"

    if not bl_path.exists():
        raise FileNotFoundError(
            f"B-L momentum for assets and rates.py introuvable: {bl_path}"
        )

    bl = _load_module_from_path("bl_momentum_runtime", bl_path)

    # Chargement local des données depuis /data
    prices = pd.read_csv(repo_root / "data" / "data_forex_prices.csv", index_col=0, parse_dates=True)
    rates_daily = pd.read_csv(repo_root / "data" / "data_fred_rates.csv", index_col=0, parse_dates=True)

    common_idx = prices.index.intersection(rates_daily.index)
    prices = prices.loc[common_idx]
    rates_daily = rates_daily.loc[common_idx]

    prices = prices.loc["2015-01-01":]
    rates_daily = rates_daily.loc["2015-01-01":]

    # Restriction aux paires demandées par l'interface
    requested_tickers = []
    for pair in pairs:
        ticker = f"{pair}=X"
        if ticker in prices.columns:
            requested_tickers.append(ticker)

    if not requested_tickers:
        raise ValueError("Aucune paire valide pour le modèle quarterly BL.")

    prices = prices[requested_tickers]

    # Recréation des returns_total et momentum_score comme dans le script BL
    returns_total = pd.DataFrame(index=prices.index, columns=requested_tickers, dtype=float)
    momentum_score = pd.DataFrame(index=prices.index, columns=requested_tickers, dtype=float)

    for t in requested_tickers:
        base, quote = bl.parse_pair(t)

        r_price = prices[t].pct_change()

        r_carry = 0.0
        if base in rates_daily.columns and quote in rates_daily.columns:
            raw_diff = rates_daily[base] - rates_daily[quote]
            realistic_diff = raw_diff - bl.BROKER_SWAP_MARKUP
            r_carry = realistic_diff / 252.0

        returns_total[t] = r_price + r_carry

        log_rets = np.log(1 + returns_total[t])
        momentum_score[t] = log_rets.rolling(bl.LOOKBACK_MOMENTUM).sum().shift(1)

    returns_total.dropna(inplace=True)
    momentum_score.dropna(inplace=True)

    start_bt = "2018-01-01"
    returns_total = returns_total.loc[start_bt:]
    momentum_score = momentum_score.loc[start_bt:]
    prices = prices.loc[start_bt:]

    if returns_total.empty or momentum_score.empty:
        raise ValueError("Pas assez de données pour calculer les poids quarterly BL.")

    # On injecte les objets globaux attendus par get_black_litterman_weights
    bl.returns_total = returns_total
    bl.momentum_score = momentum_score

    # Ajustement simple selon le niveau de risque utilisateur
    risk_map = {
        "low": 1.5,
        "medium": 2.5,
        "high": 4.0,
    }
    bl.RISK_AVERSION_DELTA = risk_map.get((risk_aversion or "").lower(), 2.5)

    last_date = returns_total.index[-1]
    raw_w = bl.get_black_litterman_weights(last_date)

    if raw_w is None or len(raw_w) == 0:
        raise ValueError("Le modèle quarterly BL a retourné des poids vides.")

    raw_weights = {
        t.replace("=X", ""): float(w)
        for t, w in zip(requested_tickers, raw_w)
    }

    final_weights = _normalize_weights(raw_weights, [t.replace("=X", "") for t in requested_tickers])

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
