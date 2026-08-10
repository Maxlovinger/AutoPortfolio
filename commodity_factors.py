"""
commodity_factors.py — build the news / fundamental / positioning / weather
SIGNAL PANELS that commodity_signals.py then tests against next-month returns.

Every factor has two parts, cleanly separated:
  * a PURE TRANSFORM (raw data -> a no-lookahead signal panel) — unit-tested;
  * a NETWORK FETCH (hit the API, cache to a pickle) — run on the user's machine.

The transforms are where the no-lookahead discipline lives:
  - tone / positioning / weather signals are standardized against each series'
    OWN TRAILING history (expanding z-score, shifted) so a value at month t uses
    only data through t.
  - inventory / crop "surprises" = actual minus a SEASONAL expectation built from
    prior years only.

Factors:
  news   (GDELT + FinBERT)  : per-commodity news tone, z-scored vs own history
  cot    (CFTC, keyless)    : speculative net positioning (hedging pressure)
  eia    (EIA API)          : petroleum / natgas inventory surprise
  weather(NOAA CDO)         : heating/cooling degree-day anomaly (natgas)
  usda   (QuickStats)       : grain stocks surprise
Column names match commodity_data (Oil, NatGas, Gold, Corn, ...).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from utils import MONTH_END

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".commodity_cache")


# ===========================================================================
# generic no-lookahead helpers (pure, tested)
# ===========================================================================
def zscore_vs_history(panel: pd.DataFrame, min_periods=12) -> pd.DataFrame:
    """
    Standardize each column against its OWN trailing history and LAG by one month:
        z_t = (x_{t-1} - mean(x_{..t-1})) / std(x_{..t-1})
    Uses an expanding window shifted by 1, so the signal available at month t is
    built only from data through t-1 (strict no-lookahead). Returns same shape.
    """
    x = panel.shift(1)                              # only past observations
    mean = x.expanding(min_periods=min_periods).mean()
    std = x.expanding(min_periods=min_periods).std(ddof=0)
    return (x - mean) / std.replace(0.0, np.nan)


def seasonal_surprise(levels: pd.Series, period=52, n_years=5) -> pd.Series:
    """
    Surprise in a (weekly) inventory/stock series = actual change minus the
    expectation formed from the SAME week in prior years only.
        change_t   = levels_t - levels_{t-1}
        expected_t = mean(change at the same seasonal slot over the last n_years,
                          strictly BEFORE t)
        surprise_t = change_t - expected_t
    Only prior-year data enters the expectation -> no lookahead.
    """
    lv = levels.dropna().sort_index()
    change = lv.diff()
    slot = np.arange(len(change)) % period
    exp = pd.Series(index=change.index, dtype=float)
    for i in range(len(change)):
        s = slot[i]
        past = change.iloc[:i][slot[:i] == s]        # same slot, strictly before i
        if len(past) >= 1:
            exp.iloc[i] = past.tail(n_years).mean()
    return (change - exp).rename(levels.name)


def degree_day_anomaly(temp: pd.Series, base=65.0, kind="HDD",
                       min_years=3) -> pd.Series:
    """
    Daily-temperature series -> MONTHLY heating/cooling degree-day ANOMALY vs the
    same calendar month's history in prior years only.
        HDD = max(0, base - T),  CDD = max(0, T - base)   (daily, summed to month)
        anomaly_t = monthly_DD_t - mean(same-month DD in prior years)
    Cold winter (HDD above normal) -> natgas-bullish -> positive anomaly.
    """
    t = temp.dropna().sort_index()
    dd = (base - t).clip(lower=0) if kind.upper() == "HDD" else (t - base).clip(lower=0)
    monthly = dd.resample(MONTH_END).sum()
    out = pd.Series(index=monthly.index, dtype=float)
    for i, ts in enumerate(monthly.index):
        prior = monthly.iloc[:i]
        same = prior[prior.index.month == ts.month]
        if len(same) >= min_years:
            out.iloc[i] = monthly.iloc[i] - same.mean()
    return out


# ===========================================================================
# 1) NEWS TONE  (GDELT + optional FinBERT), reuse news_gdelt
# ===========================================================================
# per-commodity GDELT queries (single, robust phrases — complex booleans choke
# TimelineTone, the lesson from the FX queries)
COMMODITY_QUERY = {
    "Oil": '"crude oil"', "NatGas": '"natural gas"', "Gasoline": '"gasoline prices"',
    "Gold": '"gold price"', "Silver": '"silver price"', "Copper": '"copper price"',
    "Platinum": '"platinum"', "Palladium": '"palladium"',
    "Corn": '"corn crop"', "Wheat": '"wheat prices"', "Soybeans": '"soybean"',
    "Agriculture": '"grain prices"', "Energy": '"energy prices"',
}


def commodity_tone_panel(names=None, timespan="60m", use_cache=True) -> pd.DataFrame:
    """Monthly GDELT tone per commodity -> one column each. Network; per-name
    try/except so one 429 doesn't drop the panel (same pattern as country_tone_panel)."""
    import news_gdelt as ng
    names = names or list(COMMODITY_QUERY)
    cols = {}
    print(f"Fetching commodity tone for {len(names)} markets...", flush=True)
    for i, nm in enumerate(names, 1):
        q = COMMODITY_QUERY.get(nm)
        if not q:
            continue
        try:
            s = ng.monthly_tone(q, timespan=timespan, use_cache=use_cache)
            note = f"{len(s)} pts" if not s.empty else "empty"
        except Exception as e:
            s, note = None, f"FAILED ({str(e)[:120]})"
        print(f"  [{i:>2}/{len(names)}] {nm:<12} -> {note}", flush=True)
        if s is not None and not s.empty:
            cols[nm] = s
    return pd.DataFrame(cols)


def news_signal(tone_panel: pd.DataFrame, min_periods=12) -> pd.DataFrame:
    """Tone -> no-lookahead signal: each commodity's tone z-scored vs its OWN
    trailing history (so 'unusually positive oil news' is comparable to gold's)."""
    return zscore_vs_history(tone_panel, min_periods=min_periods)


# ===========================================================================
# 2) COT POSITIONING  (CFTC, keyless Socrata)
# ===========================================================================
# commodity -> substring to match CFTC 'market_and_exchange_names'
COT_MARKET = {
    "Oil": "CRUDE OIL, LIGHT SWEET", "NatGas": "NATURAL GAS",
    "Gold": "GOLD", "Silver": "SILVER", "Copper": "COPPER",
    "Corn": "CORN", "Wheat": "WHEAT-SRW", "Soybeans": "SOYBEANS",
}
COT_RESOURCE = "6dca-aqww"          # legacy futures-only combined report


def fetch_cot(start_year=2010, use_cache=True) -> pd.DataFrame:
    """CFTC legacy futures-only report (keyless). Returns raw rows with date,
    market name, non-commercial long/short, open interest. Cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "cot_raw.pkl")
    if use_cache and os.path.exists(path):
        return pd.read_pickle(path)
    import urllib.request, json, urllib.parse
    rows, offset = [], 0
    fields = ("report_date_as_yyyy_mm_dd,market_and_exchange_names,"
              "noncomm_positions_long_all,noncomm_positions_short_all,"
              "open_interest_all")
    while True:
        q = urllib.parse.urlencode({
            "$select": fields, "$limit": 50000, "$offset": offset,
            "$where": f"report_date_as_yyyy_mm_dd >= '{start_year}-01-01T00:00:00'",
            "$order": "report_date_as_yyyy_mm_dd"})
        url = f"https://publicreporting.cftc.gov/resource/{COT_RESOURCE}.json?{q}"
        with urllib.request.urlopen(url, timeout=60) as r:
            batch = json.loads(r.read())
        if not batch:
            break
        rows += batch
        offset += 50000
        if len(batch) < 50000:
            break
    df = pd.DataFrame(rows)
    if use_cache:
        df.to_pickle(path)
    return df


def cot_positioning(raw: pd.DataFrame, market_map=None) -> pd.DataFrame:
    """
    Raw COT rows -> monthly net-speculative-position panel:
        net_spec = (noncomm_long - noncomm_short) / open_interest
    matched per commodity via the market substring, month-end last value.
    (This is the LEVEL; news_signal-style z-scoring is applied by cot_signal.)
    """
    market_map = market_map or COT_MARKET
    df = raw.copy()
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    for c in ("noncomm_positions_long_all", "noncomm_positions_short_all",
              "open_interest_all"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["net_spec"] = ((df["noncomm_positions_long_all"]
                       - df["noncomm_positions_short_all"])
                      / df["open_interest_all"].replace(0, np.nan))
    name_up = df["market_and_exchange_names"].str.upper()
    cols = {}
    for commodity, sub in market_map.items():
        m = name_up.str.contains(sub.upper(), regex=False, na=False)
        if not m.any():
            continue
        sub_df = df[m].sort_values("date")
        s = sub_df.set_index("date")["net_spec"].resample(MONTH_END).last()
        cols[commodity] = s
    return pd.DataFrame(cols)


def cot_signal(raw: pd.DataFrame, min_periods=12, market_map=None) -> pd.DataFrame:
    """Positioning -> no-lookahead signal (net-spec z-scored vs own history)."""
    return zscore_vs_history(cot_positioning(raw, market_map),
                             min_periods=min_periods)


# ===========================================================================
# 3) EIA INVENTORY SURPRISE  (EIA v2 API)
# ===========================================================================
# commodity -> (EIA v2 data route, filter) — weekly stock levels
EIA_SERIES = {
    "Oil": ("petroleum/stoc/wstk/data/", "WCESTUS1"),      # crude ex-SPR, US
    "NatGas": ("natural-gas/stor/wkly/data/", "NW2_EPG0_SWO_R48_BCF"),  # lower-48
}


def fetch_eia(series_id: str, route: str, use_cache=True) -> pd.Series:
    """Weekly level series from EIA v2. Cached. Needs EIA_API in .env."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"eia_{series_id}.pkl")
    if use_cache and os.path.exists(path):
        return pd.read_pickle(path)
    import urllib.request, json, urllib.parse
    from fx.data import load_dotenv
    load_dotenv()                                # strips surrounding quotes
    key = os.getenv("EIA_API")
    params = urllib.parse.urlencode({
        "api_key": key, "frequency": "weekly", "data[0]": "value",
        "facets[series][]": series_id, "sort[0][column]": "period",
        "sort[0][direction]": "asc", "length": 5000})
    url = f"https://api.eia.gov/v2/{route}?{params}"
    with urllib.request.urlopen(url, timeout=60) as r:
        js = json.loads(r.read())
    recs = js["response"]["data"]
    s = pd.Series({pd.Timestamp(d["period"]): float(d["value"]) for d in recs}
                  ).sort_index()
    s.name = series_id
    if use_cache:
        s.to_pickle(path)
    return s


def inventory_surprise_panel(level_series: dict) -> pd.DataFrame:
    """
    dict {commodity: weekly level Series} -> monthly surprise panel. Surprise is
    seasonal (prior-years-only); a BIGGER-than-expected DRAW is bullish, so we
    flip the sign (negative inventory change -> positive signal). Month-end mean
    of weekly surprises.
    """
    cols = {}
    for commodity, lv in level_series.items():
        surp = seasonal_surprise(lv)                 # actual change - seasonal exp
        monthly = (-surp).resample(MONTH_END).mean()  # draw (negative) -> bullish
        cols[commodity] = monthly
    return pd.DataFrame(cols)


# ===========================================================================
# 4) WEATHER  (NOAA CDO) -> natgas degree-day anomaly
# ===========================================================================
def fetch_noaa_tmax(station="GHCND:USW00094846", start="2010-01-01",
                    end=None, use_cache=True) -> pd.Series:
    """Daily average temperature (TAVG, else TMAX) for a station via NOAA CDO v2.
    Token passed in the `token:` HEADER. Cached. Needs NOAA_API in .env."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"noaa_{station.replace(':','_')}.pkl")
    if use_cache and os.path.exists(path):
        return pd.read_pickle(path)
    import urllib.request, json
    from fx.data import load_dotenv
    load_dotenv()
    token = os.getenv("NOAA_API")
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    out = {}
    # CDO caps 1yr/1000 records per call -> page by year
    for yr in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        url = ("https://www.ncei.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND"
               f"&stationid={station}&datatypeid=TMAX&units=standard"
               f"&startdate={yr}-01-01&enddate={yr}-12-31&limit=1000")
        req = urllib.request.Request(url, headers={"token": token})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                js = json.loads(r.read())
        except Exception:
            continue
        for d in js.get("results", []):
            out[pd.Timestamp(d["date"])] = float(d["value"])
    s = pd.Series(out).sort_index()
    s.name = station
    if use_cache and len(s):
        s.to_pickle(path)
    return s


def weather_signal(temp: pd.Series, kind="HDD") -> pd.DataFrame:
    """Station temps -> NatGas degree-day anomaly signal (single column)."""
    anom = degree_day_anomaly(temp, kind=kind)
    # lag one month so month-t signal uses weather realized through t-1
    return pd.DataFrame({"NatGas": anom}).shift(1)


# ===========================================================================
# assemble everything + CLI
# ===========================================================================
def build_all_signals(use_cache=True) -> dict:
    """Ingest every source (network) and return the dict of signal panels ready
    for commodity_signals.evaluate_all. Each source is wrapped so a failure in one
    (e.g. GDELT 429) doesn't sink the rest."""
    signals = {}

    def _try(name, fn):
        try:
            p = fn()
            if p is not None and len(p):
                signals[name] = p
                print(f"  built '{name}': {p.shape[1]} cols, {len(p)} months")
        except Exception as e:
            print(f"  '{name}' FAILED: {str(e)[:70]}")

    _try("news", lambda: news_signal(commodity_tone_panel(use_cache=use_cache)))
    _try("cot", lambda: cot_signal(fetch_cot(use_cache=use_cache)))
    _try("eia", lambda: inventory_surprise_panel(
        {c: fetch_eia(sid, route, use_cache) for c, (route, sid) in EIA_SERIES.items()}))
    _try("weather", lambda: weather_signal(fetch_noaa_tmax(use_cache=use_cache)))
    return signals


if __name__ == "__main__":
    import commodity_data as cd
    import commodity_signals as cs
    rets = cd.download_returns()
    rets.index = rets.index + pd.offsets.MonthEnd(0)
    print("Ingesting commodity signal sources (network; reruns use cache)...")
    signals = build_all_signals()
    if not signals:
        raise SystemExit("no signals built (check network / API keys)")
    print("\n=== SIGNAL EVALUATION vs next-month commodity returns ===")
    print("(pooled_* = time-series test, works for narrow signals; "
          "xs_* = cross-sectional, needs >=3 commodities)")
    res = cs.evaluate_all(signals, rets, n_null=300)
    cols = ["cols", "pooled_ic", "pooled_ic_z", "pooled_pctile", "ts_sharpe",
            "ts_sharpe_z", "xs_mean_ic", "xs_ic_t", "xs_ic_z", "xs_pctile",
            "n_xs", "passes"]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(res[[c for c in cols if c in res.columns]].round(3))
