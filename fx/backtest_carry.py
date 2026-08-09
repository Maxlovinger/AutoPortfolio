"""
fx/backtest_carry.py — plain carry backtest (proposal build-order step 2).

The benchmark bar every fancier layer must beat: a cross-sectional carry book.
Each rebalance we RANK currencies by their rate differential vs USD (the carry
base), go long the top sleeve and short the bottom sleeve, dollar-neutral and
equal-weight within each sleeve. No ML, no value/momentum — just carry.

Return accounting (proposal section 3): the excess return of holding foreign
currency funded in USD is
    asset_return_i = spot_return_i + (carry_i / 100) * dt
i.e. the spot move PLUS the interest differential earned over the holding
period (dt = fraction of a year). A short position reverses the sign, so the
carry cost/earn is handled automatically by the position weight.

No look-ahead: weights for the period (t, t+1] are formed from carry known at
or before t (monthly carry is forward-filled onto the trade grid then shifted).

Cadence is a parameter: freq="M" (monthly) or "W" (weekly). Weekly re-forms the
book more often — for a slow signal like carry that mostly just adds cost, which
is exactly the monthly-vs-weekly comparison we want to measure.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from utils import MONTH_END

# periods-per-year and holding fraction dt for each supported cadence
FREQ = {
    "M": {"rule": MONTH_END, "ppy": 12,  "dt": 1 / 12},
    "W": {"rule": "W-FRI",   "ppy": 52,  "dt": 1 / 52},
}


# --- Weights ---------------------------------------------------------------
def carry_weights(carry_row: pd.Series, n_long=3, n_short=3, gross=1.0) -> pd.Series:
    """
    Cross-sectional carry weights for one rebalance.
    Long the n_long highest-carry currencies, short the n_short lowest,
    equal-weight within each sleeve, dollar-neutral. Each leg carries gross/2
    of exposure so total gross = `gross`, net = 0.
    Currencies with NaN carry are ignored (can't rank them).
    """
    c = carry_row.dropna().sort_values()
    w = pd.Series(0.0, index=carry_row.index)
    if len(c) < n_long + n_short:
        # not enough names to form both sleeves this period -> stay flat
        return w
    shorts = c.index[:n_short]
    longs = c.index[-n_long:]
    w[longs] = (gross / 2) / n_long
    w[shorts] = -(gross / 2) / n_short
    return w


# --- Return panel ----------------------------------------------------------
def resample_spot(spot: pd.DataFrame, freq="M") -> pd.DataFrame:
    """Spot on the trade grid (period-end last). The grid signals compute on."""
    if freq not in FREQ:
        raise ValueError(f"freq must be one of {list(FREQ)}")
    return spot.resample(FREQ[freq]["rule"]).last()


def build_asset_returns(spot: pd.DataFrame, carry: pd.DataFrame, freq="M"):
    """
    Per-period total (carry-inclusive) asset returns on the chosen grid, plus
    the decision-time carry aligned to that grid (no look-ahead).

    Returns (asset_ret, carry_aligned), both indexed on the trade grid and
    sharing columns (the tradable currencies).
    """
    if freq not in FREQ:
        raise ValueError(f"freq must be one of {list(FREQ)}")
    cfg = FREQ[freq]

    px = spot.resample(cfg["rule"]).last()
    spot_ret = px.pct_change()

    # forward-fill monthly carry onto the grid, then shift 1 period so the
    # weight for period t only uses carry known strictly before that period.
    idx = spot_ret.index
    carry_grid = (carry.reindex(carry.index.union(idx)).ffill()
                       .reindex(idx).shift(1))

    cols = [c for c in spot_ret.columns if c in carry_grid.columns]
    spot_ret, carry_grid = spot_ret[cols], carry_grid[cols]

    asset_ret = spot_ret + (carry_grid / 100.0) * cfg["dt"]
    return asset_ret, carry_grid


# --- Backtest --------------------------------------------------------------
def run_rank_backtest(asset_ret, score_grid, freq="M", n_long=3, n_short=3,
                      gross=1.0, cost_bps=5.0):
    """
    Generic long-short cross-sectional backtest: each period, rank on
    `score_grid` (long high, short low), realize `asset_ret` net of turnover
    cost. Carry-only and the composite share this exact engine so any
    outperformance is signal, not accounting differences.

    score_grid and asset_ret must be on the same grid; the score must already
    be no-look-ahead (formed from info known before the return period).
    """
    # cost_bps may be a scalar OR a per-currency Series/dict (majors ~5bps,
    # EM ~30bps). Per-currency charges each leg's turnover at its own spread.
    cost_vec = None
    if not np.isscalar(cost_bps):
        cost_vec = pd.Series(cost_bps).reindex(asset_ret.columns).fillna(5.0)

    dates = asset_ret.index
    prev_w = pd.Series(0.0, index=asset_ret.columns)
    rows = {}
    for dt in dates:
        w = carry_weights(score_grid.loc[dt], n_long, n_short, gross)
        w = w.reindex(asset_ret.columns).fillna(0.0)
        dw = (w - prev_w).abs()
        turnover = dw.sum()
        cost = (float((dw * cost_vec).sum()) if cost_vec is not None
                else turnover * cost_bps) / 1e4
        gross_ret = float((w * asset_ret.loc[dt].fillna(0.0)).sum())
        rows[dt] = (gross_ret, gross_ret - cost, turnover)
        prev_w = w

    df = pd.DataFrame.from_dict(
        rows, orient="index", columns=["gross_ret", "net_ret", "turnover"]
    )
    # first grid row has no prior return (pct_change NaN) -> drop it
    df = df.iloc[1:]
    equity = (1 + df["net_ret"]).cumprod()
    return {
        "freq": freq,
        "net_ret": df["net_ret"],
        "gross_ret": df["gross_ret"],
        "turnover": df["turnover"],
        "equity": equity,
    }


def run_carry_backtest(spot, carry, freq="M", n_long=3, n_short=3,
                       gross=1.0, cost_bps=5.0):
    """
    Plain-carry book: rank on the carry base itself. cost_bps = half-spread on
    |Δweight| (majors ~2-5bps; EM far worse, gated separately per proposal).
    """
    asset_ret, carry_grid = build_asset_returns(spot, carry, freq)
    return run_rank_backtest(asset_ret, carry_grid, freq, n_long, n_short,
                             gross, cost_bps)


# --- Metrics ---------------------------------------------------------------
def performance(returns: pd.Series, ppy: int) -> dict:
    """
    Carry-aware performance summary. Skew and worst-period are reported
    prominently because carry's whole risk is negative skew / crash tails
    (proposal section 6/8), not volatility.
    """
    r = returns.dropna()
    if len(r) < 3:
        return {k: np.nan for k in
                ("cagr", "vol", "sharpe", "max_dd", "skew",
                 "worst", "hit_rate", "avg_turnover", "n")}
    mean, sd = r.mean(), r.std(ddof=1)
    equity = (1 + r).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    n_years = len(r) / ppy
    cagr = equity.iloc[-1] ** (1 / n_years) - 1 if equity.iloc[-1] > 0 else np.nan
    return {
        "cagr": cagr,
        "vol": sd * np.sqrt(ppy),
        "sharpe": (mean / sd) * np.sqrt(ppy) if sd > 0 else np.nan,
        "max_dd": dd,
        "skew": r.skew(),
        "worst": r.min(),
        "hit_rate": (r > 0).mean(),
        "n": len(r),
    }


def summarize(result: dict) -> dict:
    """performance() on a backtest result, tagging on average turnover."""
    ppy = FREQ[result["freq"]]["ppy"]
    m = performance(result["net_ret"], ppy)
    m["avg_turnover"] = result["turnover"].iloc[1:].mean()
    return m


# --- CLI: run monthly vs weekly on the real data ---------------------------
if __name__ == "__main__":
    from fx.data import load_all

    d = load_all(start="2010-01-01")
    spot, carry = d["spot"], d["carry"]

    print(f"{'metric':<12}{'MONTHLY':>12}{'WEEKLY':>12}")
    res = {f: run_carry_backtest(spot, carry, freq=f) for f in ("M", "W")}
    mets = {f: summarize(res[f]) for f in ("M", "W")}
    for k in ("cagr", "vol", "sharpe", "max_dd", "skew", "worst",
              "hit_rate", "avg_turnover", "n"):
        row = f"{k:<12}"
        for f in ("M", "W"):
            v = mets[f][k]
            row += f"{v:>12.3f}" if isinstance(v, float) else f"{v:>12}"
        print(row)
