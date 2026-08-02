"""
vol_forecast.py — volatility forecasts that feed the vol-targeting overlay.

The exposure decision is only as good as the volatility forecast behind it. The
original overlay used a flat trailing-std window (crude, laggy). This module
provides better, strictly non-anticipating forecasts:

  realized_vol : trailing rolling std (the original baseline).
  ewma_vol     : RiskMetrics EWMA, σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}. Reacts
                 faster to changing risk than a flat window; forecast for day t
                 uses only returns through t-1.
  vix_forecast : the ingested VIX (implied/forward-looking S&P vol), scaled down
                 by the variance-risk-premium factor (VIX runs ~structurally
                 above realized), lagged one day so it's known before the trade.
  blend        : w·VIX + (1-w)·EWMA — combines forward-looking implied vol with
                 responsive realized vol. Research finds implied-based sizing
                 gives more stable weights + lower turnover than realized alone.

ALL forecasts are aligned so forecast[t] depends only on information available
BEFORE day t's return — no look-ahead.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def realized_vol(returns: pd.Series, lookback=21) -> pd.Series:
    """Trailing annualized std, shifted so day t uses the window ending t-1."""
    return (returns.rolling(lookback).std() * np.sqrt(TRADING_DAYS)).shift(1)


def ewma_vol(returns: pd.Series, lam=0.94) -> pd.Series:
    """RiskMetrics EWMA annualized vol. forecast[t] uses returns through t-1."""
    r = returns.fillna(0.0).values
    n = len(r)
    var = np.empty(n)
    seed = np.nanvar(r[:21]) if n >= 21 else np.nanvar(r)
    prev = seed if np.isfinite(seed) and seed > 0 else float(np.var(r) or 1e-6)
    for t in range(n):
        var[t] = prev                       # forecast for t (pre-r[t])
        prev = lam * prev + (1 - lam) * r[t] ** 2
    return pd.Series(np.sqrt(var) * np.sqrt(TRADING_DAYS), index=returns.index)


def vix_forecast(returns: pd.Series, vix: pd.Series, vix_scale=0.8) -> pd.Series:
    """VIX (decimal annualized) scaled for the variance risk premium, lagged 1d."""
    vx = (vix * vix_scale).reindex(returns.index).ffill().shift(1)
    return vx


def blend(returns: pd.Series, vix: pd.Series, lam=0.94, vix_scale=0.8,
          w=0.5) -> pd.Series:
    """Weighted blend of VIX-implied and EWMA-realized forecasts."""
    ew = ewma_vol(returns, lam=lam)
    vx = vix_forecast(returns, vix, vix_scale=vix_scale)
    f = w * vx + (1 - w) * ew
    return f.fillna(ew)                      # fall back to EWMA where VIX missing


def load_vix(path="vix.pkl") -> pd.Series:
    return pd.read_pickle(path)


# ----------------------------------------------------------------------
# Overlay: turn a vol forecast into an exposure-scaled return series
# ----------------------------------------------------------------------
def exposure(forecast: pd.Series, target=0.15, max_lev=1.0) -> pd.Series:
    """Target/forecast, capped at max_lev (no leverage by default)."""
    e = (target / forecast).clip(upper=max_lev)
    return e.fillna(max_lev)


def apply_overlay(returns: pd.Series, forecast: pd.Series, target=0.15,
                  max_lev=1.0):
    """Return (overlaid_returns, exposure_series). forecast must be
    non-anticipating (day t known before r[t])."""
    e = exposure(forecast, target=target, max_lev=max_lev).reindex(returns.index)
    e = e.fillna(max_lev)
    return returns * e, e


def make_forecast(returns: pd.Series, method="ewma", vix=None, **kw) -> pd.Series:
    """Dispatch helper."""
    if method == "realized":
        return realized_vol(returns, **{k: v for k, v in kw.items()
                                        if k == "lookback"})
    if method == "ewma":
        return ewma_vol(returns, **{k: v for k, v in kw.items() if k == "lam"})
    if method == "vix":
        return vix_forecast(returns, vix, **{k: v for k, v in kw.items()
                                             if k == "vix_scale"})
    if method == "blend":
        return blend(returns, vix, **{k: v for k, v in kw.items()
                                      if k in ("lam", "vix_scale", "w")})
    raise ValueError(f"unknown method {method}")
