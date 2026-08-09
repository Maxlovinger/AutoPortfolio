"""
foreign_data.py — Tier-1 data layer for the foreign-equity strategy.

Tier 1 answers ONE question cheaply, before any single-stock engine is built:
does foreign equity actually diversify the US-equity + carry book? That question
needs only broad/country INDEX returns, which come free from yfinance as ETF
proxies. No point-in-time single-stock constituents, no survivorship headache.

All ETFs here are US-listed and priced in USD, so `auto_adjust` close already
gives a USD total return (price + dividends). The broad/country funds are
UNHEDGED (their USD NAV = local stock move + FX move) — exactly the "bundled"
exposure the design doc describes. A few currency-HEDGED funds are included so we
can measure what hedging removes (they strip the FX leg via forwards internally).

  developed_*  : ex-US developed (EAFE) — expected ~0.8 corr to US (redundant)
  em_*         : emerging markets — the real diversifier candidate
  hedged       : currency-hedged versions, to isolate the FX contribution
  EM_CARRY_ETF : EM carry currency -> its single-country equity ETF, for the
                 shared-tail check (does EM equity crash WITH EM carry?)

Returns are cached to foreign_prices.pkl so the gate reruns instantly offline.
"""
from __future__ import annotations
import os
import pandas as pd

CACHE = os.path.join(os.path.dirname(__file__), "foreign_prices.pkl")

# --- universe (all US-listed, USD) -----------------------------------------
DEVELOPED_BROAD = {"EAFE": "EFA", "DevExUS": "VEA"}
DEVELOPED_COUNTRY = {
    "Japan": "EWJ", "Germany": "EWG", "UK": "EWU", "France": "EWQ",
    "Switzerland": "EWL", "Canada": "EWC", "Australia": "EWA",
}
EM_BROAD = {"EM": "EEM", "EM_vanguard": "VWO"}
EM_COUNTRY = {
    "Brazil": "EWZ", "India": "INDA", "China": "FXI", "Taiwan": "EWT",
    "SouthKorea": "EWY", "Mexico": "EWW", "SouthAfrica": "EZA",
    "Chile": "ECH", "Poland": "EPOL", "Israel": "EIS",
}
# currency-hedged developed funds (isolate FX): (unhedged, hedged) share a market
HEDGED = {"EAFE_hedged": "HEFA", "Japan_hedged": "HEWJ", "Eurozone_hedged": "HEZU"}
HEDGE_PAIRS = [("EAFE", "EAFE_hedged"), ("Japan", "Japan_hedged")]

US_BENCH = {"US": "SPY"}

# EM carry currency -> matching single-country equity ETF (for shared-tail test).
# CZK and HUF have no liquid single-country US ETF, so they're absent here.
EM_CARRY_ETF = {
    "MXN": "Mexico", "PLN": "Poland", "KRW": "SouthKorea",
    "CLP": "Chile", "ILS": "Israel", "ZAR": "SouthAfrica",
}

# name -> ticker over the whole universe
ALL = {**{k: v for k, v in US_BENCH.items()},
       **DEVELOPED_BROAD, **DEVELOPED_COUNTRY,
       **EM_BROAD, **EM_COUNTRY, **HEDGED}


def all_tickers() -> list[str]:
    return sorted(set(ALL.values()))


# --- download / cache ------------------------------------------------------
def _monthly_total_return(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end adjusted close -> monthly total return (USD)."""
    from utils import MONTH_END
    px = prices.resample(MONTH_END).last()
    return px.pct_change()


def download_returns(start="2005-01-01", end=None, use_cache=True,
                     refresh=False) -> pd.DataFrame:
    """
    Monthly USD total returns for every ETF proxy, columns = friendly NAMES
    (not tickers). Cached to disk; pass refresh=True to force a re-download.

    Needs network (yfinance/Yahoo). On a blocked host, run once on a networked
    machine to populate foreign_prices.pkl, then everything downstream is offline.
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
    # ticker -> friendly name (invert ALL)
    inv = {v: k for k, v in ALL.items()}
    ret = ret.rename(columns=inv)
    ret = ret[[c for c in ALL if c in ret.columns]]   # stable, named order

    if use_cache:
        ret.to_pickle(CACHE)
    return ret


if __name__ == "__main__":
    r = download_returns(refresh=True)
    print(f"Downloaded {r.shape[1]} proxies, "
          f"{r.index[0]:%Y-%m}..{r.index[-1]:%Y-%m} ({len(r)} months)")
    cov = r.notna().sum().sort_values()
    print("\nmonths of history per proxy (short = recent-inception fund):")
    for name, n in cov.items():
        print(f"  {name:<15}{n:>4}")
    print(f"\ncached -> {CACHE}")
