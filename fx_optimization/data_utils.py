"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  CHARGEMENT DES DONNÉES                                                        ║
║  • data_forex_prices.csv — prix de clôture FX (Yahoo)                          ║
║  • data_fred_rates.csv   — taux directeurs par devise (FRED)                   ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

from config import TRADING_DAYS, FILE_PRICES, FILE_RATES, TICKER_TO_PAIR, USER
from utils_display import hdr, tbl


def parse_pair(ticker: str) -> Tuple[str, str]:
    t = ticker.replace("=X", "")
    return t[:3], t[3:6]


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(FILE_PRICES, index_col=0, parse_dates=True).sort_index()
    rates  = pd.read_csv(FILE_RATES,  index_col=0, parse_dates=True).sort_index()
    start = max(prices.index.min(), rates.index.min())
    end   = min(prices.index.max(), rates.index.max())
    prices = prices.loc[start:end].ffill()
    rates  = rates.loc[start:end].ffill()
    return prices, rates


def get_fx_universe(prices: pd.DataFrame, rates: pd.DataFrame) -> List[str]:
    rate_ccys = set(rates.columns)
    valid = []
    for ticker in prices.columns:
        if not ticker.endswith("=X"):
            continue
        base, quote = parse_pair(ticker)
        if base in rate_ccys and quote in rate_ccys:
            # Si l'utilisateur a choisi des paires spécifiques
            pair_name = TICKER_TO_PAIR.get(ticker, "")
            if USER.selected_pairs is not None:
                if pair_name not in USER.selected_pairs:
                    continue
            valid.append(ticker)
    return valid


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    ret = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    return ret.dropna(how="all").dropna(axis=0, how="any")


def compute_carry(pairs: List[str], rates: pd.DataFrame) -> pd.DataFrame:
    carry = pd.DataFrame(index=rates.index)
    for t in pairs:
        base, quote = parse_pair(t)
        carry[t] = (rates[base] - rates[quote]) / TRADING_DAYS
    return carry


def ticker_to_pair_name(ticker: str) -> str:
    return TICKER_TO_PAIR.get(ticker, ticker.replace("=X", ""))