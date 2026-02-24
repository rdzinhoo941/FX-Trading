from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_pair(ticker: str) -> Tuple[str, str]:
    """Yahoo FX tickers are of the form 'EURUSD=X' (base=EUR, quote=USD)."""
    t = ticker.replace("=X", "")
    if len(t) < 6:
        raise ValueError(f"Unrecognized FX ticker: {ticker}")
    return t[:3], t[3:6]


def load_fx_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df.ffill()
    return df


def load_rates(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["DATE"]).set_index("DATE").sort_index()
    return df


def intersect_universe_with_rates(fx_prices: pd.DataFrame, rates: pd.DataFrame) -> List[str]:
    """Keep only FX tickers for which both currencies have rate proxies in rates CSV."""
    rate_ccy = set(rates.columns)
    ok = []
    for t in fx_prices.columns:
        base, quote = parse_pair(t)
        if base in rate_ccy and quote in rate_ccy:
            ok.append(t)
    return ok


def compute_returns(fx_prices: pd.DataFrame) -> pd.DataFrame:
    """Simple returns (pct_change)."""
    ret = fx_prices.pct_change().replace([np.inf, -np.inf], np.nan)
    ret = ret.dropna(how="all").dropna(axis=0, how="any")
    return ret


def compute_carry(pairs: List[str], rates_aligned: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """
    Carry proxy for FX: (r_base - r_quote)/252.
    Interpretable as the (approx.) expected daily forward premium under covered interest parity.
    """
    carry = pd.DataFrame(index=rates_aligned.index)
    for t in pairs:
        base, quote = parse_pair(t)
        diff_annual = rates_aligned[base] - rates_aligned[quote]
        carry[t] = diff_annual / trading_days
    return carry


# -----------------------------
# Entropy Pooling (fast solver)
# -----------------------------
def entropy_pooling_moment_tilt(
    R: np.ndarray,
    mu_target: np.ndarray,
    p: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    maxiter: int = 100,
    ridge: float = 1e-10,
) -> Tuple[np.ndarray, bool, str]:
    """
    Solve: min_q KL(q||p) s.t. sum(q)=1 and R^T q = mu_target (moment constraints only).
    Exponential tilting solution: q_i ∝ p_i exp(-R_i·b). Choose b s.t. E_q[R]=mu_target.
    Uses damped Newton iterations on g(b)=E_q[R]-mu_target.
    """
    S, I = R.shape
    if p is None:
        p = np.full(S, 1.0 / S)
    p = np.asarray(p, dtype=float)
    p = p / p.sum()

    b = np.zeros(I)

    def compute_q(bvec: np.ndarray) -> Optional[np.ndarray]:
        z = -R @ bvec
        z = z - np.max(z)  # stabilize exponentials
        w = p * np.exp(z)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            return None
        return w / s

    q = compute_q(b)
    if q is None:
        return p.copy(), False, "numerical issue in initialization"

    mu_q = R.T @ q
    best_err = float(np.linalg.norm(mu_q - mu_target))
    best_q = q.copy()
    best_b = b.copy()

    for it in range(maxiter):
        mu_q = R.T @ q
        diff = mu_q - mu_target
        err = float(np.linalg.norm(diff))
        if err < tol:
            return q, True, f"converged in {it} iters"
        if err < best_err:
            best_err, best_q, best_b = err, q.copy(), b.copy()

        # Cov_q(R)
        X = R - mu_q
        Cov = (X.T * q) @ X
        Cov = Cov + ridge * np.eye(I)

        # Newton step: b_new = b + Cov^{-1} * diff  (since dE/db = -Cov)
        try:
            step = np.linalg.solve(Cov, diff)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(Cov, diff, rcond=None)[0]

        alpha = 1.0
        improved = False
        for _ in range(20):
            b_new = b + alpha * step
            q_new = compute_q(b_new)
            if q_new is None:
                alpha *= 0.5
                continue
            mu_new = R.T @ q_new
            err_new = float(np.linalg.norm(mu_new - mu_target))
            if err_new < err:
                b, q = b_new, q_new
                improved = True
                break
            alpha *= 0.5

        if not improved:
            break

    return best_q, False, f"did not fully converge; best error {best_err:.3e}"


def posterior_moments(R: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = R.T @ q
    X = R - mu
    Sigma = (X.T * q) @ X
    return mu, Sigma


def tangency_weights(
    mu: np.ndarray,
    Sigma: np.ndarray,
    gross_target: float = 1.0,
    w_max: float = 0.25,
    ridge: float = 1e-6,
) -> np.ndarray:
    """
    Tangency (max Sharpe) direction: w ∝ Sigma^{-1} mu.
    Then normalize to a fixed gross exposure and cap per-asset weight.
    """
    I = len(mu)
    Sigma_reg = Sigma + ridge * np.eye(I)
    w = np.linalg.pinv(Sigma_reg) @ mu
    if np.allclose(w, 0):
        return np.zeros(I)

    gross = float(np.sum(np.abs(w)))
    if gross > 0:
        w = w * (gross_target / gross)

    if w_max is not None:
        w = np.clip(w, -w_max, w_max)
        gross = float(np.sum(np.abs(w)))
        if gross > 0:
            w = w * (gross_target / gross)
    return w


# -----------------------------
# Backtest + performance
# -----------------------------
def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def perf_metrics(returns: pd.Series, rf_daily: Optional[pd.Series] = None, trading_days: int = 252) -> Dict[str, float]:
    r = returns.dropna()
    if rf_daily is None:
        rf_daily = pd.Series(0.0, index=r.index)
    rf_daily = rf_daily.reindex(r.index).ffill().fillna(0.0)
    excess = r - rf_daily

    nav = (1.0 + r).cumprod()
    n = len(r)

    cagr = float(nav.iloc[-1] ** (trading_days / n) - 1.0)
    ann_vol = float(excess.std(ddof=0) * math.sqrt(trading_days))
    sharpe = float((excess.mean() * trading_days) / (excess.std(ddof=0) * math.sqrt(trading_days) + 1e-12))
    mdd = max_drawdown(nav)
    calmar = float(cagr / (abs(mdd) + 1e-12))

    return {
        "FinalNAV": float(nav.iloc[-1]),
        "TotalReturn": float(nav.iloc[-1] - 1.0),
        "CAGR": cagr,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": mdd,
        "Calmar": calmar,
        "NObs": float(n),
    }


@dataclass
class BacktestResult:
    metrics: Dict[str, float]
    trades: int
    rebalances: int
    avg_turnover: float
    nav: pd.Series
    daily_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series


def backtest_entropy_pooling(
    ret: pd.DataFrame,
    carry: pd.DataFrame,
    rf_daily: pd.Series,
    lookback: int = 252,
    rebalance_every: int = 21,
    view_strength: float = 0.25,
    tc_bps_per_leg: float = 5.0,
    gross_target: float = 1.0,
    w_max: float = 0.25,
    target_vol_annual: Optional[float] = 0.10,
    leverage_max: float = 2.0,
    min_obs: int = 252,
    eps_trade: float = 1e-6,
    trading_days: int = 252,
) -> BacktestResult:
    """
    Rolling EP + tangency portfolio backtest.

    Transaction costs model:
      cost_t = (2 * tc_bps_per_leg / 1e4) * turnover
      where turnover = sum_i |w_t - w_{t-1}|.
    """
    dates = ret.index
    assets = list(ret.columns)
    I = len(assets)

    w_prev = np.zeros(I)
    weights_hist = []
    turnover_series = pd.Series(index=dates, dtype=float)
    port_ret = pd.Series(index=dates, dtype=float)

    trades = 0
    rebalances = 0

    for t_idx, dt in enumerate(dates):
        if t_idx < min_obs:
            continue

        if (t_idx - min_obs) % rebalance_every == 0:
            # build scenario matrix on rolling window
            window = slice(max(0, t_idx - lookback), t_idx)
            R = ret.iloc[window].to_numpy()
            if R.shape[0] < 50:
                continue

            mu_prior = R.mean(axis=0)
            carry_mean = carry.iloc[window].mean(axis=0).to_numpy()

            # view: blend historical mean and carry
            mu_view = (1.0 - view_strength) * mu_prior + view_strength * carry_mean

            # feasibility guard: clip to per-asset scenario range
            mu_view = np.minimum(mu_view, R.max(axis=0))
            mu_view = np.maximum(mu_view, R.min(axis=0))

            q, ok, msg = entropy_pooling_moment_tilt(R, mu_view)
            mu_post, Sigma_post = posterior_moments(R, q)

            w = tangency_weights(mu_post, Sigma_post, gross_target=gross_target, w_max=w_max)

            # volatility targeting (optional)
            if target_vol_annual is not None:
                vol_daily = float(np.sqrt(max(w.T @ Sigma_post @ w, 0.0)))
                vol_ann = vol_daily * math.sqrt(trading_days)
                if vol_ann > 1e-10:
                    lev = target_vol_annual / vol_ann
                    lev = min(lev, leverage_max)
                    w = w * lev

            delta = w - w_prev
            turnover = float(np.sum(np.abs(delta)))
            tc = (2.0 * tc_bps_per_leg / 1e4) * turnover

            # cost is charged on rebalance day
            port_ret.loc[dt] = float(w_prev @ ret.iloc[t_idx].to_numpy()) - tc
            turnover_series.loc[dt] = turnover

            trades += int(np.sum(np.abs(delta) > eps_trade))
            rebalances += 1

            w_prev = w
            weights_hist.append((dt, *w.tolist(), ok, msg))
        else:
            port_ret.loc[dt] = float(w_prev @ ret.iloc[t_idx].to_numpy())

    port_ret = port_ret.dropna()
    rf_use = rf_daily.reindex(port_ret.index).ffill().fillna(0.0)
    metrics = perf_metrics(port_ret, rf_use, trading_days=trading_days)
    nav = (1.0 + port_ret).cumprod()

    weights_df = pd.DataFrame(weights_hist, columns=["Date"] + assets + ["EP_hard_ok", "EP_msg"]).set_index("Date")
    avg_turn = float(turnover_series.dropna().mean()) if turnover_series.notna().any() else 0.0

    return BacktestResult(
        metrics=metrics,
        trades=trades,
        rebalances=rebalances,
        avg_turnover=avg_turn,
        nav=nav,
        daily_returns=port_ret,
        weights=weights_df,
        turnover=turnover_series.dropna(),
    )


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Entropy Pooling FX strategy backtest")
    ap.add_argument("--fx_csv", type=str, default="data_forex_prices.csv")
    ap.add_argument("--rates_csv", type=str, default="data_fred_rates.csv")

    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--rebalance_every", type=int, default=21)
    ap.add_argument("--view_strength", type=float, default=0.25)

    ap.add_argument("--tc_bps_per_leg", type=float, default=5.0, help="Transaction cost in bps PER LEG (round-trip ≈ 2x).")
    ap.add_argument("--gross_target", type=float, default=1.0)
    ap.add_argument("--w_max", type=float, default=0.25)

    ap.add_argument("--target_vol_annual", type=float, default=0.10)
    ap.add_argument("--leverage_max", type=float, default=2.0)

    ap.add_argument("--out_prefix", type=str, default="ep_fx")
    args = ap.parse_args()

    fx = load_fx_prices(args.fx_csv)
    rates = load_rates(args.rates_csv)
    pairs = intersect_universe_with_rates(fx, rates)
    if len(pairs) < 2:
        raise RuntimeError("Universe too small after intersecting with available rates columns.")

    fx = fx[pairs]
    # align rates to FX dates
    rates_aligned = rates.reindex(fx.index).ffill()

    ret = compute_returns(fx)
    carry = compute_carry(pairs, rates_aligned).reindex(ret.index).ffill()
    rf_daily = (rates_aligned["USD"] / 252.0).reindex(ret.index).ffill().fillna(0.0)

    res = backtest_entropy_pooling(
        ret=ret,
        carry=carry,
        rf_daily=rf_daily,
        lookback=args.lookback,
        rebalance_every=args.rebalance_every,
        view_strength=args.view_strength,
        tc_bps_per_leg=args.tc_bps_per_leg,
        gross_target=args.gross_target,
        w_max=args.w_max,
        target_vol_annual=args.target_vol_annual,
        leverage_max=args.leverage_max,
    )

    print("\n========== Entropy Pooling FX Strategy ==========")
    print(f"Universe size: {len(pairs)}")
    print(f"Lookback: {args.lookback} | Rebalance every: {args.rebalance_every} days")
    print(f"View strength: {args.view_strength}")
    print(f"Transaction costs: {args.tc_bps_per_leg} bps per leg")
    print(f"Trades (approx): {res.trades} | Rebalances: {res.rebalances} | Avg turnover: {res.avg_turnover:.3f}")
    print("-------------------------------------------------")
    for k, v in res.metrics.items():
        if k == "NObs":
            print(f"{k:>10s}: {int(v)}")
        elif k == "Sharpe":
            print(f"{k:>10s}: {v:.3f}")
        elif k == "FinalNAV":
            print(f"{k:>10s}: {v:.4f}")
        else:
            # returns/vol/dd ratios as percentages
            print(f"{k:>10s}: {v:.4%}")
    print("=================================================\n")

    # save outputs
    res.nav.to_frame("NAV").to_csv(f"{args.out_prefix}_nav.csv")
    res.daily_returns.to_frame("ret").to_csv(f"{args.out_prefix}_daily_returns.csv")
    res.weights.to_csv(f"{args.out_prefix}_weights.csv")
    res.turnover.to_frame("turnover").to_csv(f"{args.out_prefix}_turnover.csv")


if __name__ == "__main__":
    main()
