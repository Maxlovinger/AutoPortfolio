"""
news_gdelt.py — historical news-tone backbone (GDELT 2.0 DOC API).

GDELT ships a PRE-COMPUTED average "tone" for any query over time, timestamped
at publication — which is what makes it BACKTESTABLE (no look-ahead), unlike the
live yfinance/NewsAPI feeds that only give "now". We use the TimelineTone mode:
one tone point per period for a search query, then aggregate to month-end so it
aligns with the equity/FX monthly grids.

Two entry points:
  * entity_tone(name)   -> company/organization tone   (equity sleeve)
  * country_tone(ccy)   -> country/economy tone         (FX sleeve)

Operational notes:
  * GDELT rate-limits HARD and blocks datacenter IPs (we saw persistent HTTP 429
    from the sandbox). From a normal/residential IP it works; we add polite
    backoff + an on-disk cache so a backtest doesn't re-hit the API.
  * No API key required.
  * Tone is roughly [-10, +10] (negative..positive); we rescale to ~[-1, 1] to
    match the FinBERT/VADER convention before use.
"""
from __future__ import annotations
import json
import os
import time
import hashlib
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".gdelt_cache")

# currency -> a query that isolates that economy's news (FX sleeve)
CCY_QUERY = {
    "EUR": '"euro zone" OR "European Central Bank" OR eurozone economy',
    "GBP": '"United Kingdom" economy OR "Bank of England"',
    "JPY": 'Japan economy OR "Bank of Japan"',
    "CHF": 'Switzerland economy OR "Swiss National Bank"',
    "AUD": 'Australia economy OR "Reserve Bank of Australia"',
    "NZD": '"New Zealand" economy OR "Reserve Bank of New Zealand"',
    "CAD": 'Canada economy OR "Bank of Canada"',
    "SEK": 'Sweden economy OR Riksbank',
    "NOK": 'Norway economy OR Norges Bank',
    "USD": '"United States" economy OR "Federal Reserve"',
}


def _cache_path(params: dict) -> str:
    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.json")


def _fetch(params: dict, retries=4, pause=5.0, use_cache=True) -> dict:
    """GET the DOC API with backoff + on-disk cache. Raises on repeated 429."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = _cache_path(params)
    if use_cache and os.path.exists(cp):
        with open(cp) as fh:
            return json.load(fh)

    url = DOC_URL + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
            data = json.loads(body)
            with open(cp, "w") as fh:
                json.dump(data, fh)
            return data
        except Exception as e:                       # 429 / transient / JSON
            last = e
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"GDELT fetch failed after {retries} tries: {last}")


def _timeline_to_series(data: dict) -> pd.Series:
    """Parse a TimelineTone response into a tone Series (rescaled to ~[-1,1])."""
    tl = data.get("timeline") or []
    if not tl:
        return pd.Series(dtype=float)
    pts = tl[0].get("data", [])
    idx = pd.to_datetime([p["date"] for p in pts])
    vals = np.array([p["value"] for p in pts], dtype=float) / 10.0
    return pd.Series(np.clip(vals, -1, 1), index=idx).sort_index()


def tone_timeline(query: str, timespan="24m", use_cache=True) -> pd.Series:
    """Raw (sub-daily) tone timeline for an arbitrary GDELT query, ~[-1,1]."""
    params = {"query": query, "mode": "TimelineTone", "format": "json",
              "timespan": timespan}
    return _timeline_to_series(_fetch(params, use_cache=use_cache))


def monthly_tone(query: str, timespan="24m", use_cache=True) -> pd.Series:
    """Month-end mean tone for a query — aligns with the monthly trade grids."""
    s = tone_timeline(query, timespan=timespan, use_cache=use_cache)
    if s.empty:
        return s
    return s.resample("M").mean()


def entity_tone(name: str, **kw) -> pd.Series:
    """Company/organization monthly tone (equity sleeve). `name` = company name
    (news matches names better than tickers)."""
    return monthly_tone(f'"{name}"', **kw)


def country_tone(ccy: str, **kw) -> pd.Series:
    """Country/economy monthly tone for a currency code (FX sleeve)."""
    if ccy not in CCY_QUERY:
        raise KeyError(f"no GDELT query mapped for {ccy}")
    return monthly_tone(CCY_QUERY[ccy], **kw)


def country_tone_panel(ccys, **kw) -> pd.DataFrame:
    """Monthly tone for several currencies -> one column each (FX sentiment)."""
    cols = {}
    for c in ccys:
        s = country_tone(c, **kw)
        if not s.empty:
            cols[c] = s
    return pd.DataFrame(cols)
