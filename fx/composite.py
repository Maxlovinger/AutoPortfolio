"""
fx/composite.py — carry + value + momentum ranker (proposal build-order step 3).

Three documented FX style premia (proposal section 4C), each turned into a
cross-sectional signal, z-scored across currencies each period, and averaged
into one composite score that the SAME rank engine trades:

  * CARRY     : rate differential vs USD (the step-2 signal).
  * MOMENTUM  : trailing spot return (winners keep winning, ~12m in FX).
  * VALUE     : PPP-lite mean reversion — the NEGATIVE long-horizon (~5y) spot
                return. A currency that has fallen a long way is "cheap" and
                expected to revert up. This is the Asness-Moskowitz-Pedersen
                value proxy used when full CPI-based PPP data isn't on hand;
                upgrading to CPI-based real-rate PPP is a later refinement.

No look-ahead: every signal is computed from period-end spot then shifted one
period, exactly like the carry alignment, so weights for (t, t+1] use only
info known by t. Returns are still carry-inclusive (carry is part of realized
P&L no matter what we rank on).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from fx.backtest_carry import (
    FREQ, resample_spot, build_asset_returns, run_rank_backtest,
)


# --- Individual signals (each returned on the trade grid, no look-ahead) ---
def _months_to_periods(months, freq):
    return max(1, round(months / 12 * FREQ[freq]["ppy"]))


def momentum_signal(spot, freq="M", lookback_months=12):
    """Trailing spot return over lookback_months, shifted 1 period."""
    px = resample_spot(spot, freq)
    k = _months_to_periods(lookback_months, freq)
    return px.pct_change(k).shift(1)


def value_signal(spot, freq="M", years=5):
    """PPP-lite: negative long-horizon spot return (cheap = fell = high score)."""
    px = resample_spot(spot, freq)
    k = _months_to_periods(years * 12, freq)
    return -px.pct_change(k).shift(1)


def carry_signal(spot, carry, freq="M"):
    """Carry base aligned to the grid (reuses build_asset_returns' shift)."""
    _, carry_grid = build_asset_returns(spot, carry, freq)
    return carry_grid


# --- Combination -----------------------------------------------------------
def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional (row-wise) z-score, NaN-safe. Rows with <2 valid names
    or zero dispersion return zeros so they don't dominate the average."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=0)
    z = df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    return z


def composite_score(spot, carry, freq="M", weights=None,
                    mom_months=12, value_years=5):
    """
    Average of z-scored carry, momentum, value on the shared grid.
    `weights` = dict over {"carry","momentum","value"} (default equal). Signals
    that are NaN for a currency/period (e.g. no 5y history yet) are skipped in
    that cell's average, so early periods still trade on whatever is available.
    """
    weights = weights or {"carry": 1.0, "momentum": 1.0, "value": 1.0}
    parts = {
        "carry": zscore_rows(carry_signal(spot, carry, freq)),
        "momentum": zscore_rows(momentum_signal(spot, freq, mom_months)),
        "value": zscore_rows(value_signal(spot, freq, value_years)),
    }
    # weighted nan-mean across the active signals, cell by cell
    num = None
    den = None
    for name, z in parts.items():
        w = weights.get(name, 0.0)
        if w == 0:
            continue
        contrib = (z * w).fillna(0.0)
        mask = z.notna().astype(float) * w
        num = contrib if num is None else num + contrib
        den = mask if den is None else den + mask
    score = num.div(den.replace(0, np.nan))
    return score


# --- Backtest --------------------------------------------------------------
def run_composite_backtest(spot, carry, freq="M", n_long=3, n_short=3,
                           gross=1.0, cost_bps=5.0, weights=None,
                           mom_months=12, value_years=5):
    """Trade the composite score on the shared rank engine (carry-inclusive
    returns), so it's directly comparable to run_carry_backtest."""
    asset_ret, _ = build_asset_returns(spot, carry, freq)
    score = composite_score(spot, carry, freq, weights, mom_months, value_years)
    score = score.reindex(asset_ret.index)[asset_ret.columns]
    return run_rank_backtest(asset_ret, score, freq, n_long, n_short,
                             gross, cost_bps)


# --- CLI: does the composite beat plain carry? -----------------------------
if __name__ == "__main__":
    from fx.data import load_all
    from fx.backtest_carry import run_carry_backtest, summarize

    d = load_all(start="2010-01-01")
    spot, carry = d["spot"], d["carry"]
    freq = "M"

    books = {
        "carry": run_carry_backtest(spot, carry, freq=freq),
        "mom+val+carry": run_composite_backtest(spot, carry, freq=freq),
        "carry+mom": run_composite_backtest(
            spot, carry, freq=freq, weights={"carry": 1, "momentum": 1, "value": 0}),
        "carry+val": run_composite_backtest(
            spot, carry, freq=freq, weights={"carry": 1, "momentum": 0, "value": 1}),
    }
    mets = {k: summarize(v) for k, v in books.items()}

    keys = ["cagr", "vol", "sharpe", "max_dd", "skew", "worst",
            "hit_rate", "avg_turnover", "n"]
    hdr = f"{'metric':<12}" + "".join(f"{k:>15}" for k in books)
    print(hdr)
    for m in keys:
        row = f"{m:<12}"
        for k in books:
            v = mets[k][m]
            row += f"{v:>15.3f}" if isinstance(v, float) else f"{v:>15}"
        print(row)
