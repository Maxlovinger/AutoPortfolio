"""
commodity_data.py — Tier-1 data layer for the commodity strategy.

Same cheap-gate-first philosophy as foreign_data.py: before touching real
futures data (roll-adjusted continuous contracts — the hard, paid part), answer
one question with free ETF proxies — does commodity exposure, and specifically a
trend-following commodity BOOK, diversify the equity + carry portfolio?

Commodity ETFs hold futures, so their USD total return already bakes in the
ROLL YIELD (the term-structure carry) plus the spot move — exactly the return a
long futures position earns. That makes them a fair Tier-1 proxy for passive
commodity exposure and for a trend overlay. What they CANNOT cleanly isolate is
pure cross-sectional carry (needs the futures curve) — that waits for Tier 2.

  BROAD    : diversified baskets (DBC, GSG)
  ENERGY / METALS / AGS : single-sector / single-commodity funds
  TREND_BOOK : the subset used to build the time-series-momentum strategy —
               single commodities spanning 4 sectors with long history.

Cached to commodity_prices.pkl so the gate reruns offline.
"""
from __future__ import annotations
import os
import pandas as pd

CACHE = os.path.join(os.path.dirname(__file__), "commodity_prices.pkl")

# --- universe (all US-listed futures-backed ETFs, USD) ---------------------
BROAD = {"Commodities": "DBC", "GSCI": "GSG"}
ENERGY = {"Oil": "USO", "NatGas": "UNG", "Energy": "DBE", "Gasoline": "UGA"}
METALS = {"Gold": "GLD", "Silver": "SLV", "BaseMetals": "DBB",
          "Copper": "CPER", "Platinum": "PPLT", "Palladium": "PALL"}
AGS = {"Agriculture": "DBA", "Corn": "CORN", "Wheat": "WEAT",
       "Soybeans": "SOYB"}
US_BENCH = {"US": "SPY"}

# single commodities spanning energy/precious/base/ags, all ~2007+ history,
# used to build the trend-following book (need breadth, not the broad baskets)
TREND_BOOK = ["Oil", "NatGas", "Gold", "Silver", "BaseMetals", "Agriculture"]

COMMODITY_GROUPS = {
    "broad": list(BROAD),
    "energy": list(ENERGY),
    "metals": list(METALS),
    "ags": list(AGS),
}

ALL = {**US_BENCH, **BROAD, **ENERGY, **METALS, **AGS}


def all_tickers() -> list[str]:
    return sorted(set(ALL.values()))


# --- download / cache ------------------------------------------------------
def _monthly_total_return(prices: pd.DataFrame) -> pd.DataFrame:
    from utils import MONTH_END
    px = prices.resample(MONTH_END).last()
    return px.pct_change()


def download_returns(start="2006-01-01", end=None, use_cache=True,
                     refresh=False) -> pd.DataFrame:
    """
    Monthly USD total returns per commodity ETF, columns = friendly NAMES.
    Cached; refresh=True forces a re-download. Needs network (yfinance).
    """
    if use_cache and not refresh and os.path.exists(CACHE):
        cached = pd.read_pickle(CACHE)
        if start is not None:
            cached = cached[cached.index >= pd.Timestamp(start)]
        return cached

    import yfinance as yf
    tickers = all_tickers()
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)
    prices = raw["Close"] if "Close" in raw else raw
    prices = prices.dropna(how="all").ffill()

    ret = _monthly_total_return(prices)
    inv = {v: k for k, v in ALL.items()}
    ret = ret.rename(columns=inv)
    ret = ret[[c for c in ALL if c in ret.columns]]

    if use_cache:
        ret.to_pickle(CACHE)
    return ret


if __name__ == "__main__":
    r = download_returns(refresh=True)
    print(f"Downloaded {r.shape[1]} proxies, "
          f"{r.index[0]:%Y-%m}..{r.index[-1]:%Y-%m} ({len(r)} months)")
    print("\nmonths of history per proxy:")
    for name, n in r.notna().sum().sort_values().items():
        print(f"  {name:<14}{n:>4}")
    print(f"\ncached -> {CACHE}")
