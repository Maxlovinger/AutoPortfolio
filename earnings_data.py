"""
earnings_data.py — historical earnings-surprise panel (the PEAD signal source).

Post-Earnings-Announcement Drift (PEAD): names that beat estimates tend to drift
up for weeks after; misses drift down. The driver is the SURPRISE itself (a hard
number), so we don't need NLP for the core signal — just actual-vs-estimate EPS.

Source: yfinance `get_earnings_dates` — free, ~15y deep, gives EPS Estimate,
Reported EPS, and Surprise(%) per announcement. (Alpha Vantage EARNINGS is an
alternative/cross-check but its free tier is 25 req/day; yfinance has no such
cap.) Announcement dates are tz-localized to naive so they align with prices.

The panel is SPARSE: Surprise(%) sits on each announcement date, NaN elsewhere.
Downstream (earnings_signal.py) reads "most recent surprise within the drift
window" as of each rebalance — strictly no look-ahead (earnings after the
decision date are unknown).
"""
from __future__ import annotations
import os
import pandas as pd
import yfinance as yf

CACHE = os.path.join(os.path.dirname(__file__), "earnings_surprise.pkl")


def fetch_surprises(tickers, limit=60, cache=CACHE, refresh=False) -> pd.DataFrame:
    """Sparse panel of Surprise(%) — index = announcement dates (tz-naive),
    columns = tickers. Cached; only missing tickers are fetched on rerun."""
    have = {}
    if cache and os.path.exists(cache) and not refresh:
        panel = pd.read_pickle(cache)
        have = {c: panel[c].dropna() for c in panel.columns}
    missing = [t for t in tickers if t not in have]
    n = len(missing)
    if n:
        print(f"Fetching earnings surprises for {n} names via yfinance...",
              flush=True)
    for i, t in enumerate(missing, 1):
        try:
            df = yf.Ticker(t).get_earnings_dates(limit=limit)
            s = df["Surprise(%)"].dropna()
            idx = pd.to_datetime(s.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)          # align with tz-naive prices
            s = pd.Series(s.values, index=idx)
            s = s[~s.index.duplicated()].sort_index()
            have[t] = s
            note = f"{len(s)} earnings, {s.index.min().date()}..{s.index.max().date()}"
        except Exception as e:
            note = f"FAILED ({str(e)[:30]})"
        print(f"  [{i:>3}/{n}] {t:<6} -> {note}", flush=True)
        if cache and i % 20 == 0:                     # incremental save
            pd.DataFrame(have).sort_index().to_pickle(cache)
    panel = pd.DataFrame(have).sort_index()
    if cache and not panel.empty:
        panel.to_pickle(cache)
    return panel


if __name__ == "__main__":
    p = fetch_surprises(["AAPL", "MU", "NVDA", "JPM"])
    print(p.tail(8).round(2))
