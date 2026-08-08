"""
fx/data.py — data layer for the currency model (proposal steps 1-2).

Two free sources, mirroring the equity system's reliance on yfinance:
  * SPOT FX      -> yfinance (Yahoo).  Daily close, majors only, no bid/ask.
  * SHORT RATES  -> FRED (needs FRED_API in ../.env).  Used to build carry.

Design notes that matter more than the code:
  * Everything is expressed as "value of 1 unit of the FOREIGN currency in USD".
    So a RISE in the series = the foreign currency strengthened vs USD, and a
    long position in that currency profits. yfinance quotes some pairs the other
    way (USDJPY = JPY per USD), so those are inverted here — see SPOT below.
  * CARRY BASE = foreign short rate - USD short rate. Covered interest parity
    says this ~= the forward premium, so we don't need a paid forwards feed to
    get a usable carry signal (proposal section 3/7).
  * The OECD 3M interbank series (IR3TIB01...) are MONTHLY and LAG by 1-7 months.
    Fine for a monthly-rebalanced backtest; too stale for live trading. USD has
    a fresh daily fallback (DGS3MO) but we use the OECD USD series by default so
    all legs are apples-to-apples in the backtest.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

# --- Universe config -------------------------------------------------------
# For each currency vs USD:
#   fred   : verified 3M interbank rate series (monthly, OECD)
#   ticker : yfinance spot pair
#   invert : True if the ticker quotes USD-per-foreign the wrong way
#            (i.e. it's FOREIGN per USD and we must take 1/price)
G10 = {
    "USD": {"fred": "IR3TIB01USM156N", "ticker": None,       "invert": False},
    "EUR": {"fred": "IR3TIB01EZM156N", "ticker": "EURUSD=X", "invert": False},
    "GBP": {"fred": "IR3TIB01GBM156N", "ticker": "GBPUSD=X", "invert": False},
    "JPY": {"fred": "IR3TIB01JPM156N", "ticker": "USDJPY=X", "invert": True},
    "CHF": {"fred": "IR3TIB01CHM156N", "ticker": "USDCHF=X", "invert": True},
    "AUD": {"fred": "IR3TIB01AUM156N", "ticker": "AUDUSD=X", "invert": False},
    "NZD": {"fred": "IR3TIB01NZM156N", "ticker": "NZDUSD=X", "invert": False},
    "CAD": {"fred": "IR3TIB01CAM156N", "ticker": "USDCAD=X", "invert": True},
    "SEK": {"fred": "IR3TIB01SEM156N", "ticker": "USDSEK=X", "invert": True},
    "NOK": {"fred": "IR3TIB01NOM156N", "ticker": "USDNOK=X", "invert": True},
}

BASE = "USD"  # rates are measured as a differential vs this


def _fred_client() -> Fred:
    """Build a FRED client from FRED_API in ../.env (or the environment)."""
    load_dotenv()  # reads .env from the project root if present
    key = os.getenv("FRED_API")
    if not key:
        raise RuntimeError(
            "FRED_API not found. Put FRED_API=<key> in the project .env file."
        )
    return Fred(api_key=key)


# --- Short rates -----------------------------------------------------------
def fetch_short_rates(start="2000-01-01", universe=G10) -> pd.DataFrame:
    """Monthly 3M interbank rate (in %) per currency. One column per currency."""
    fred = _fred_client()
    cols = {}
    for ccy, cfg in universe.items():
        s = fred.get_series(cfg["fred"], observation_start=start).dropna()
        cols[ccy] = s
    rates = pd.DataFrame(cols)
    # Normalize to month-end so it aligns cleanly with resampled spot.
    rates.index = pd.to_datetime(rates.index)
    rates = rates.resample("M").last()
    return rates


def carry_base(rates: pd.DataFrame, base=BASE) -> pd.DataFrame:
    """
    Rate differential vs the base currency, in % (proposal's carry anchor).
    Positive = higher-yielding than USD = you get paid to be long it.
    The base column itself becomes ~0 and is dropped.
    """
    diff = rates.sub(rates[base], axis=0)
    return diff.drop(columns=[base])


# --- Spot FX ---------------------------------------------------------------
def fetch_spot(start="2000-01-01", end=None, universe=G10) -> pd.DataFrame:
    """
    Daily 'USD value of 1 foreign unit', one column per non-USD currency.
    Inverts the pairs that Yahoo quotes as foreign-per-USD so every column
    has the same orientation (up = foreign strengthened).
    """
    tickers = [c["ticker"] for c in universe.values() if c["ticker"]]
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    px = raw["Close"] if "Close" in raw else raw
    px = px.ffill()

    out = {}
    for ccy, cfg in universe.items():
        if not cfg["ticker"]:
            continue
        col = px[cfg["ticker"]]
        out[ccy] = 1.0 / col if cfg["invert"] else col
    spot = pd.DataFrame(out).dropna(how="all")
    return spot


def monthly_fx_returns(spot: pd.DataFrame) -> pd.DataFrame:
    """Month-end simple returns of the USD-per-foreign spot (the spot-return
    leg of total FX excess return; carry is added on top later)."""
    m = spot.resample("M").last()
    return m.pct_change().dropna(how="all")


# --- Convenience -----------------------------------------------------------
def load_all(start="2000-01-01", end=None, universe=G10):
    """Fetch everything the first carry backtest needs.

    Returns dict with:
        rates       : monthly short rates (%)
        carry       : monthly rate differential vs USD (%), the carry signal
        spot        : daily USD-per-foreign spot
        fx_returns  : monthly spot returns
    """
    rates = fetch_short_rates(start=start, universe=universe)
    spot = fetch_spot(start=start, end=end, universe=universe)
    return {
        "rates": rates,
        "carry": carry_base(rates, base=BASE),
        "spot": spot,
        "fx_returns": monthly_fx_returns(spot),
    }


if __name__ == "__main__":
    d = load_all(start="2010-01-01")
    print("=== short rates (last 3 months) ===")
    print(d["rates"].tail(3).round(2))
    print("\n=== carry base vs USD (last 3 months, %) ===")
    print(d["carry"].tail(3).round(2))
    print("\n=== spot USD-per-foreign (last 3 rows) ===")
    print(d["spot"].tail(3).round(4))
    print("\n=== monthly FX returns (last 3 months) ===")
    print(d["fx_returns"].tail(3).round(4))
