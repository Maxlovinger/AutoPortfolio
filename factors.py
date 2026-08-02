"""
factors.py — (A) The value + momentum + quality composite, with interaction.

For each stock we compute three classic factor scores, z-score them across the
universe, and combine. The "interaction" term rewards stocks that are cheap
AND improving (value x momentum) — the part most screeners skip.

Data: yfinance .info (fundamentals) + price history (momentum).
Everything is wrapped so missing fundamentals never kill the pipeline.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

from utils import zscore, winsorize, safe


def _fundamentals(ticker: str) -> dict:
    """Pull the fundamental fields we need for one ticker."""
    info = safe(lambda: yf.Ticker(ticker).info, default={}) or {}
    return {
        "pe":   info.get("trailingPE", np.nan),
        "pb":   info.get("priceToBook", np.nan),
        "ev_ebitda": info.get("enterpriseToEbitda", np.nan),
        "roe":  info.get("returnOnEquity", np.nan),
        "margin": info.get("profitMargins", np.nan),
        "d2e":  info.get("debtToEquity", np.nan),
    }


def momentum_scores(prices: pd.DataFrame) -> pd.Series:
    """
    12-1 momentum: total return over the past 12 months, EXCLUDING the most
    recent month (classic academic definition — skips short-term reversal).
    """
    if len(prices) < 252:
        window = prices
    else:
        window = prices.iloc[-252:-21]  # ~12mo ago .. ~1mo ago
    mom = window.iloc[-1] / window.iloc[0] - 1.0
    return mom


def build_factor_table(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Return a DataFrame indexed by ticker with raw + z-scored factors."""
    rows = {t: _fundamentals(t) for t in tickers}
    f = pd.DataFrame(rows).T

    # ---- VALUE: cheap = high score, so we NEGATE the ratios ----
    value_raw = -pd.concat(
        [zscore(winsorize(f["pe"])),
         zscore(winsorize(f["pb"])),
         zscore(winsorize(f["ev_ebitda"]))],
        axis=1,
    ).mean(axis=1)

    # ---- QUALITY: high ROE / margin, low leverage ----
    quality_raw = pd.concat(
        [zscore(winsorize(f["roe"])),
         zscore(winsorize(f["margin"])),
         -zscore(winsorize(f["d2e"]))],
        axis=1,
    ).mean(axis=1)

    # ---- MOMENTUM ----
    mom = momentum_scores(prices).reindex(tickers)
    mom_raw = zscore(winsorize(mom))

    out = pd.DataFrame({
        "value": value_raw.reindex(tickers).fillna(0.0),
        "quality": quality_raw.reindex(tickers).fillna(0.0),
        "momentum": mom_raw.reindex(tickers).fillna(0.0),
    })

    # ---- INTERACTION: cheap AND rising (the differentiated bit) ----
    out["value_x_mom"] = zscore(out["value"] * out["momentum"])
    return out


def composite_score(ftable: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Weighted blend of the factor columns -> single alpha score per ticker."""
    w = weights or {"value": 1.0, "quality": 1.0, "momentum": 1.0, "value_x_mom": 0.5}
    score = sum(ftable[k] * wt for k, wt in w.items() if k in ftable)
    return zscore(score).sort_values(ascending=False)
