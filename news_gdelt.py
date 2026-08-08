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

from utils import MONTH_END

DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".gdelt_cache")
# GDELT rate-limits ~1 req / few sec; a polite delay AFTER each live fetch avoids
# tripping 429s (which otherwise force slow backoffs). Cache hits don't sleep.
# 3s spacing cut the 429 failure rate materially in testing; rerun to fill gaps
# (cached names return instantly, only the failed ones retry).
THROTTLE_SEC = 3.0

# currency -> a SINGLE quoted phrase (GDELT TimelineTone chokes on complex boolean
# OR queries; single phrases like the equity path work). Central-bank tone is also
# the most FX-relevant news signal — policy/risk sentiment drives currencies.
CCY_QUERY = {
    "EUR": '"European Central Bank"',
    "GBP": '"Bank of England"',
    "JPY": '"Bank of Japan"',
    "CHF": '"Swiss National Bank"',
    "AUD": '"Reserve Bank of Australia"',
    "NZD": '"Reserve Bank of New Zealand"',
    "CAD": '"Bank of Canada"',
    "SEK": '"Sveriges Riksbank"',
    "NOK": '"Norges Bank"',
    "USD": '"Federal Reserve"',
}


def _cache_path(params: dict) -> str:
    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.json")


def _fetch(params: dict, retries=3, pause=2.0, use_cache=True,
           throttle=THROTTLE_SEC) -> dict:
    """GET the DOC API with backoff + on-disk cache. Raises on repeated failure.
    Cache HITS return immediately (no sleep); after a live fetch we pause
    `throttle`s to stay under GDELT's rate limit."""
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
            time.sleep(throttle)                     # polite spacing after a hit
            return data
        except Exception as e:                       # 429 / transient / JSON
            last = e
            time.sleep(pause * (attempt + 1))        # 2s, 4s, 6s then give up
    raise RuntimeError(f"GDELT fetch failed after {retries} tries: {last}")


def _timeline_to_series(data: dict) -> pd.Series:
    """Parse a TimelineTone response into a tone Series (rescaled to ~[-1,1])."""
    tl = data.get("timeline") or []
    if not tl:
        return pd.Series(dtype=float)
    pts = tl[0].get("data", [])
    idx = pd.to_datetime([p["date"] for p in pts])
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)          # GDELT stamps are UTC; drop tz so it
                                             # aligns with tz-naive price/rate data
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
    return s.resample(MONTH_END).mean()


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
    """Monthly tone for several currencies -> one column each (FX sentiment).
    Per-currency error handling so one 429 doesn't drop the whole panel; prints
    progress + the actual date span (reruns resume from the per-query cache)."""
    cols = {}
    n = len(ccys)
    print(f"Fetching country tone for {n} economies "
          f"(reruns resume from .gdelt_cache/)...", flush=True)
    for i, c in enumerate(ccys, 1):
        try:
            s = country_tone(c, **kw)
            note = (f"{len(s)} pts, {s.index.min().date()}..{s.index.max().date()}"
                    if not s.empty else "empty")
        except Exception as e:
            s, note = None, f"FAILED ({str(e)[:34]})"
        print(f"  [{i:>2}/{n}] {c:<4} -> {note}", flush=True)
        if s is not None and not s.empty:
            cols[c] = s
    return pd.DataFrame(cols)
