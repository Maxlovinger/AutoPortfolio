"""
costs.py — realistic per-name transaction cost model for the backtester.

WHY
---
A flat basis-point cost (e.g. 10bps everywhere) badly understates the friction
of trading small-caps, which is exactly where naive backtests hide fake alpha
(high turnover in thin names looks free but isn't). Real one-way cost has two
parts:

  1. HALF-SPREAD  : you cross the bid-ask. Wider for illiquid / low-priced names.
                    Modeled as a decreasing function of dollar ADV.
  2. MARKET IMPACT: your own order moves the price. Grows with how much of a
                    day's volume you consume, via the square-root ("Almgren")
                    law:  impact ∝ sqrt(trade$ / ADV$).

Both are driven by ADV (average daily dollar volume), so the model needs a
per-name ADV series and a portfolio capital figure (impact scales with size —
a $10M book pushes prices more than a $100k book).

CALIBRATION (defaults, one-way):
  half-spread:  ~$1M ADV → 25bps,  ~$100M → 2.5bps,  ~$1B → 0.8bps
  impact:       trading 1% of a day's volume → ~10bps,  10% → ~32bps
These are deliberately conservative for small-caps; tune via kwargs.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def half_spread_bps(adv_usd, *, base=25.0, floor=1.0, ceil=75.0):
    """Half the bid-ask spread in bps as a function of dollar ADV.

    base is the half-spread (bps) at $1M ADV; it decays like 1/sqrt(ADV) and is
    clipped to [floor, ceil] so mega-caps aren't free and penny names aren't
    absurd."""
    adv = np.asarray(adv_usd, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        bps = base / np.sqrt(np.maximum(adv, 1.0) / 1e6)
    return np.clip(bps, floor, ceil)


def impact_bps(participation, *, coef=100.0):
    """Square-root market-impact law, in bps.

    participation = trade$ / ADV$ (fraction of a day's volume consumed).
    coef≈100 → 1% participation ≈ 10bps, 10% ≈ 32bps, 100% ≈ 100bps."""
    p = np.maximum(np.asarray(participation, dtype=float), 0.0)
    return coef * np.sqrt(p)


def trade_cost_fraction(trade_value, adv_usd, *, base=25.0, floor=1.0,
                        ceil=75.0, coef=100.0):
    """One-way cost as a FRACTION of the traded notional, per name.

    trade_value : dollar value traded (sign ignored)
    adv_usd     : dollar ADV for that name
    """
    tv = np.abs(np.asarray(trade_value, dtype=float))
    adv = np.maximum(np.asarray(adv_usd, dtype=float), 1.0)
    hs = half_spread_bps(adv, base=base, floor=floor, ceil=ceil)
    im = impact_bps(tv / adv, coef=coef)
    return (hs + im) / 1e4


def rebalance_cost(dweights: pd.Series, adv_usd: pd.Series, capital: float,
                   **kw) -> float:
    """Total cost of one rebalance as a FRACTION of NAV.

    dweights : weight CHANGES per name this rebalance (target - current)
    adv_usd  : dollar ADV per name (missing names fall back to the median)
    capital  : portfolio NAV in dollars — sets the market-impact scale

    Cost-of-NAV = Σ_i |Δw_i| · cost_fraction_i, since trading |Δw_i| of NAV at
    cost_fraction_i per traded dollar costs |Δw_i|·cost_fraction_i of NAV.
    """
    dw = dweights[dweights.abs() > 0]
    if dw.empty:
        return 0.0
    fallback = float(adv_usd.median()) if len(adv_usd) else 1e6
    adv = adv_usd.reindex(dw.index).fillna(fallback)
    trade_value = dw.abs() * float(capital)
    frac = trade_cost_fraction(trade_value.values, adv.values, **kw)
    return float((dw.abs().values * frac).sum())


def load_adv(path="universe.csv") -> pd.Series:
    """Dollar ADV per ticker from the eligibility file (adv_usd column)."""
    df = pd.read_csv(path, index_col=0)
    col = "adv_usd" if "adv_usd" in df.columns else df.columns[0]
    return pd.to_numeric(df[col], errors="coerce").dropna()
