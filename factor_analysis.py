"""
factor_analysis.py — measure whether each factor actually predicts returns.

Two standard quant diagnostics:

1. INFORMATION COEFFICIENT (IC)
   For each date, the cross-sectional rank correlation between a factor's values
   today and each stock's forward return. Averaged over time it answers: "does a
   higher factor value reliably precede a higher return?"
     mean IC  ~ predictive strength   (equities: 0.02-0.05 is genuinely useful)
     IC IR    = mean(IC)/std(IC) annualized  ~ consistency (the key number)
     t-stat   ~ statistical significance
     hit rate ~ fraction of periods the factor pointed the right way

2. QUANTILE ANALYSIS
   Each period, sort stocks into N buckets by factor value and average each
   bucket's forward return. A GOOD factor is MONOTONIC (Q5 > Q4 > ... > Q1) with
   a positive long-short spread (top bucket minus bottom bucket).

Both use non-overlapping sampling (sample = forward horizon) so the statistics
aren't inflated by autocorrelated overlapping returns.

POINT-IN-TIME: only price-derived factors are analysed — they can be correctly
reconstructed from past prices. Fundamentals/sentiment need a PIT database.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# Forward returns & factor panels (all point-in-time safe)
# ----------------------------------------------------------------------
def forward_returns(prices: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Return over the NEXT `horizon` days for each stock at each date."""
    return prices.shift(-horizon) / prices - 1.0


def factor_panels(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Historical value of each candidate factor at every date (dates x tickers).
    Every factor is computed from PAST prices only.
    Sign convention: higher factor value is hypothesised to be BETTER, so
    volatility (lower is better) is negated.
    """
    logret = np.log(prices / prices.shift(1))
    out = {
        "momentum_12_1": prices.shift(21) / prices.shift(252) - 1,   # classic
        "mom_21":  prices / prices.shift(21) - 1,
        "mom_126": prices / prices.shift(126) - 1,
        "mom_252": prices / prices.shift(252) - 1,
        "low_vol_21": -logret.rolling(21).std(),      # negated: low vol = high score
        "low_vol_63": -logret.rolling(63).std(),
        "reversal":  -(prices / prices.rolling(21).mean() - 1),
        "dist_high": prices / prices.rolling(252).max() - 1,
    }
    return out


# ----------------------------------------------------------------------
# Information Coefficient
# ----------------------------------------------------------------------
def ic_series(factor: pd.DataFrame, fwd: pd.DataFrame,
              method: str = "spearman", sample: int = 21,
              min_names: int = 5) -> pd.Series:
    """Time series of cross-sectional IC, sampled every `sample` days."""
    common = factor.index.intersection(fwd.index)
    dates = common[::sample]
    out = {}
    for t in dates:
        f, r = factor.loc[t], fwd.loc[t]
        valid = f.notna() & r.notna()
        if valid.sum() < min_names:
            continue
        ic = f[valid].corr(r[valid], method=method)
        if np.isfinite(ic):
            out[t] = ic
    return pd.Series(out, name="IC")


def ic_summary(ic: pd.Series, horizon: int = 21) -> dict:
    """Summary stats from an IC time series."""
    ic = ic.dropna()
    n = len(ic)
    if n < 2:
        return {k: float("nan") for k in
                ("mean_ic", "ic_std", "ic_ir", "t_stat", "hit_rate", "n")}
    mean, sd = ic.mean(), ic.std(ddof=1)
    periods_per_year = TRADING_DAYS / horizon
    ic_ir = (mean / sd) * np.sqrt(periods_per_year) if sd > 0 else float("nan")
    t_stat = (mean / (sd / np.sqrt(n))) if sd > 0 else float("nan")
    return {"mean_ic": mean, "ic_std": sd, "ic_ir": ic_ir,
            "t_stat": t_stat, "hit_rate": (ic > 0).mean(), "n": n}


# ----------------------------------------------------------------------
# Quantile analysis
# ----------------------------------------------------------------------
def quantile_returns(factor: pd.DataFrame, fwd: pd.DataFrame,
                     n_q: int = 5, sample: int = 21,
                     min_names: int = 5) -> pd.Series:
    """
    Average forward return per factor quantile (1=lowest .. n_q=highest), plus a
    'long_short' entry (top minus bottom). Returns are per-`sample`-period.
    """
    common = factor.index.intersection(fwd.index)
    buckets = {q: [] for q in range(1, n_q + 1)}
    for t in common[::sample]:
        f, r = factor.loc[t], fwd.loc[t]
        valid = f.notna() & r.notna()
        if valid.sum() < max(min_names, n_q):
            continue
        try:
            q = pd.qcut(f[valid], n_q, labels=False, duplicates="drop") + 1
        except Exception:
            continue
        grp = r[valid].groupby(q).mean()
        for k, v in grp.items():
            buckets[int(k)].append(v)
    means = {q: (np.mean(vals) if vals else np.nan) for q, vals in buckets.items()}
    s = pd.Series(means).sort_index()
    s.index = [f"Q{q}" for q in s.index]
    s["long_short"] = s.get(f"Q{n_q}", np.nan) - s.get("Q1", np.nan)
    return s


def is_monotonic(qret: pd.Series, n_q: int = 5) -> bool:
    """True if quantile mean returns increase monotonically Q1..Qn."""
    vals = [qret.get(f"Q{q}", np.nan) for q in range(1, n_q + 1)]
    vals = [v for v in vals if np.isfinite(v)]
    return len(vals) >= 2 and all(x < y for x, y in zip(vals, vals[1:]))


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def analyze(prices: pd.DataFrame, horizon: int = 21, n_q: int = 5):
    """Run IC + quantile analysis for every factor. Returns (summary_df, quantiles_dict, ic_dict)."""
    fwd = forward_returns(prices, horizon)
    panels = factor_panels(prices)
    rows, quantiles, ics = [], {}, {}
    for name, fac in panels.items():
        ic = ic_series(fac, fwd, sample=horizon)
        summ = ic_summary(ic, horizon)
        qr = quantile_returns(fac, fwd, n_q=n_q, sample=horizon)
        summ["name"] = name
        summ["monotonic"] = is_monotonic(qr, n_q)
        summ["long_short"] = qr.get("long_short", np.nan)
        rows.append(summ)
        quantiles[name] = qr
        ics[name] = ic
    summary = pd.DataFrame(rows).set_index("name")
    summary = summary.sort_values("ic_ir", ascending=False,
                                  key=lambda s: s.abs())
    return summary, quantiles, ics


def main():
    from data import download_prices
    from screener import DEFAULT_UNIVERSE
    prices = download_prices(DEFAULT_UNIVERSE)
    print(f"Factor IC/quantile analysis on {prices.shape[1]} stocks, "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}\n")
    summary, quantiles, _ = analyze(prices)

    cols = ["mean_ic", "ic_ir", "t_stat", "hit_rate", "long_short", "monotonic", "n"]
    show = summary[cols].copy()
    for c in ["mean_ic", "long_short"]:
        show[c] = (show[c] * 100).round(2)
    show[["ic_ir", "t_stat"]] = show[["ic_ir", "t_stat"]].round(2)
    show["hit_rate"] = (show["hit_rate"] * 100).round(0)
    pd.set_option("display.width", 200)
    print("Ranked by |IC IR| (annualized consistency of the signal):\n")
    print(show)
    print("\nGuide: mean_ic & long_short in %.  |IC IR|>0.5 and |t|>2 = worth "
          "keeping.  monotonic=True means clean quantile ordering.")
    return summary, quantiles


if __name__ == "__main__":
    main()
