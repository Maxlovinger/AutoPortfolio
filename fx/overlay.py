"""
fx/overlay.py — crash-risk / vol-target overlay (proposal build-order step 4).

Carry's problem in our tests isn't weak signal, it's the negative-skew crash
tail (monthly plain carry: skew -0.35, DD -6.9%). This overlay scales the whole
book's exposure by the ratio of a target vol to the book's OWN trailing realized
vol, cutting exposure as volatility rises. Because carry crashes ARE volatility
spikes (safe-haven rushes), that de-risks *ahead of / into* the crash.

Why realized-vol (not VIX/implied): on the equity side of this project realized-
vol targeting repeatedly beat VIX breakers and fancier schemes, and the BoJ
(2020) FX microstructure paper shows spreads blow out and liquidity vanishes
exactly during stress — so you want to be OUT before you're forced to trade into
a widening spread. Realized book vol turns up as the book itself starts hurting.

Everything is causal: exposure for period t uses vol of returns through t-1
only (rolling window then shift 1). Changing exposure costs |Δexposure| * spread
(approximated on the dollar-neutral gross=1 book).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from fx.backtest_carry import FREQ


def _months_to_periods(months, freq):
    return max(1, round(months / 12 * FREQ[freq]["ppy"]))


def realized_vol(returns: pd.Series, lookback: int, ppy: int) -> pd.Series:
    """Annualized trailing realized vol over `lookback` periods."""
    return returns.rolling(lookback).std(ddof=1) * np.sqrt(ppy)


def vol_target_exposure(returns, target_vol, freq="M", lookback_months=6,
                        cap=3.0, warmup=1.0) -> pd.Series:
    """
    Causal exposure multiplier k_t = target_vol / realized_vol_{t-1}, capped at
    `cap` (leverage limit). Before enough history for a vol estimate, run at
    `warmup` exposure (default 1.0 = the raw book). Never goes short the book
    (k >= 0).
    """
    ppy = FREQ[freq]["ppy"]
    k_lb = _months_to_periods(lookback_months, freq)
    rv = realized_vol(returns, k_lb, ppy).shift(1)      # <- no look-ahead
    k = (target_vol / rv).clip(lower=0.0, upper=cap)
    return k.fillna(warmup)


def apply_overlay(returns: pd.Series, exposure: pd.Series, cost_bps=5.0):
    """
    Scale returns by exposure, charging |Δexposure| * half-spread for the
    resizing. Returns (overlaid_net_returns, exposure_used).
    """
    k = exposure.reindex(returns.index).fillna(0.0)
    resize = k.diff().abs()
    resize.iloc[0] = abs(k.iloc[0])                     # establishing exposure
    cost = resize * cost_bps / 1e4
    net = k * returns - cost
    return net, k


def run_vol_target(result: dict, target_vol=0.05, lookback_months=6,
                   cap=3.0, cost_bps=5.0) -> dict:
    """
    Apply the vol-target overlay to a backtest result (from run_carry_backtest /
    run_composite_backtest). Returns a result dict in the same shape, plus the
    exposure path, so it drops straight into summarize().
    """
    freq = result["freq"]
    r = result["net_ret"]
    exp = vol_target_exposure(r, target_vol, freq, lookback_months, cap)
    net, k = apply_overlay(r, exp, cost_bps)
    net = net.dropna()
    return {
        "freq": freq,
        "net_ret": net,
        "gross_ret": result.get("gross_ret"),
        "turnover": result["turnover"],
        "exposure": k.reindex(net.index),
        "equity": (1 + net).cumprod(),
    }


# --- CLI: does the overlay improve plain carry? ----------------------------
if __name__ == "__main__":
    from fx.data import load_all
    from fx.backtest_carry import run_carry_backtest, summarize

    d = load_all(start="2010-01-01")
    spot, carry = d["spot"], d["carry"]
    freq = "M"

    base = run_carry_backtest(spot, carry, freq=freq)
    books = {"carry (raw)": base}
    for tv in (0.03, 0.05, 0.07):
        books[f"carry+VT{int(tv*100)}"] = run_vol_target(
            base, target_vol=tv, lookback_months=6, cap=3.0)
    mets = {k: summarize(v) for k, v in books.items()}

    keys = ["cagr", "vol", "sharpe", "max_dd", "skew", "worst", "hit_rate", "n"]
    print(f"{'metric':<10}" + "".join(f"{k:>14}" for k in books))
    for m in keys:
        row = f"{m:<10}"
        for k in books:
            v = mets[k][m]
            row += f"{v:>14.3f}" if isinstance(v, float) else f"{v:>14}"
        print(row)
    # show how hard the overlay leans
    for k, v in books.items():
        if "exposure" in v:
            e = v["exposure"]
            print(f"{k}: avg exposure {e.mean():.2f}, min {e.min():.2f}, "
                  f"max {e.max():.2f}")
