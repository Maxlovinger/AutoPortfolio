"""
historical_membership.py — point-in-time S&P index membership from Wikipedia.

WHAT THIS FIXES (and what it does NOT)
--------------------------------------
`universe.py` uses *today's* constituents, so any historical backtest on it is
survivorship-biased: it only ever sees companies that survived to the present.
This module reconstructs *who was actually in the index on each past date*,
INCLUDING names that were later removed / delisted / acquired — the free
"membership half" of the survivorship fix (the Robot Wealth method).

METHOD
------
For each index page (S&P 500 / 400 / 600) Wikipedia provides:
  * the CURRENT constituents table, and
  * a CHANGES table (effective date, ticker added, ticker removed, reason).
Starting from today's set we walk the changes list BACKWARD in time: to get the
membership just before a change we UNDO it — drop the added ticker, restore the
removed ticker. Repeating this yields point-in-time snapshots back to ~2000
(accuracy degrades before then; Wikipedia's changes log is "selected").

THE CATCH — PRICES
------------------
Knowing a delisted name was a member is only half the job; a backtest also needs
its PRICE history, and yfinance drops most delisted symbols. So this makes the
UNIVERSE honest but leaves a PRICE gap — the bias is reduced, not eliminated.
`price_coverage()` quantifies exactly how many historical members Yahoo can still
serve. Closing the gap fully needs a paid vendor (Norgate/Sharadar).
"""
from __future__ import annotations
import io
import urllib.request
import numpy as np
import pandas as pd

from universe import clean_ticker

SP_PAGES = {
    "SP500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "SP400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "SP600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}


# ----------------------------------------------------------------------
# Scraping / parsing
# ----------------------------------------------------------------------
def _read_tables(url: str) -> list[pd.DataFrame]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "ignore")
    return pd.read_html(io.StringIO(html))


def _flatten(col) -> str:
    return "/".join(str(x) for x in col) if isinstance(col, tuple) else str(col)


def parse_current(tables: list[pd.DataFrame]) -> set[str]:
    """Current constituents: the first table with a Symbol/Ticker column."""
    for t in tables:
        cols = {_flatten(c).lower(): c for c in t.columns}
        sym = next((cols[k] for k in cols
                    if "symbol" in k or k.endswith("ticker")), None)
        # skip the changes table (only it has a "removed" column)
        if sym is None or any("removed" in k for k in cols):
            continue
        return {clean_ticker(x) for x in t[sym].dropna() if str(x).strip()}
    raise ValueError("No current-constituents table found")


def parse_changes(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Return tidy changes: columns [date, added, removed]. Empty if none."""
    for t in tables:
        flat = {_flatten(c).lower(): c for c in t.columns}
        has_add = any("added" in k and "ticker" in k for k in flat)
        has_rem = any("removed" in k and "ticker" in k for k in flat)
        if not (has_add and has_rem):
            continue
        datecol = next((flat[k] for k in flat if "date" in k), None)
        addcol = next(flat[k] for k in flat if "added" in k and "ticker" in k)
        remcol = next(flat[k] for k in flat if "removed" in k and "ticker" in k)
        raw_date = (t[datecol].astype(str).str.replace(r"\[.*?\]", "", regex=True)
                    if datecol is not None else pd.Series([""] * len(t)))
        out = pd.DataFrame({
            "date": pd.to_datetime(raw_date, errors="coerce"),
            "added": t[addcol].map(_clean_opt),
            "removed": t[remcol].map(_clean_opt),
        })
        out = out.dropna(subset=["date"])
        out = out[(out["added"] != "") | (out["removed"] != "")]
        return out.sort_values("date").reset_index(drop=True)
    return pd.DataFrame(columns=["date", "added", "removed"])


def _clean_opt(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip()
    return "" if s.lower() in ("", "nan") else clean_ticker(s)


# ----------------------------------------------------------------------
# Reconstruction (pure — fully testable offline)
# ----------------------------------------------------------------------
def members_on(date, current: set[str], changes: pd.DataFrame) -> set[str]:
    """
    Point-in-time membership as of `date`, by rewinding every change that took
    effect AFTER `date` (newest first): undo add = drop it, undo remove = restore.
    """
    members = set(current)
    date = pd.Timestamp(date)
    fut = changes[changes["date"] > date].sort_values("date", ascending=False)
    for _, row in fut.iterrows():
        if row["added"]:
            members.discard(row["added"])
        if row["removed"]:
            members.add(row["removed"])
    return members


def membership_snapshots(current: set[str], changes: pd.DataFrame,
                         start="2010-01-01", end=None, freq="MS") -> pd.DataFrame:
    """
    Long-format point-in-time membership: columns [date, ticker], one row per
    (month-start, member). `freq` is a pandas offset ('MS' = month start).
    """
    end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
    dates = pd.date_range(start=start, end=end, freq=freq)
    rows = []
    for d in dates:
        for tk in members_on(d, current, changes):
            rows.append((d, tk))
    return pd.DataFrame(rows, columns=["date", "ticker"])


# ----------------------------------------------------------------------
# End-to-end build + persistence
# ----------------------------------------------------------------------
def build_membership(indices=("SP500", "SP400", "SP600"),
                     start="2010-01-01", freq="MS",
                     out="membership.csv", verbose=True) -> pd.DataFrame:
    """Scrape all indices, reconstruct PIT membership, save long CSV."""
    frames = []
    for key in indices:
        tables = _read_tables(SP_PAGES[key])
        current = parse_current(tables)
        changes = parse_changes(tables)
        snap = membership_snapshots(current, changes, start=start, freq=freq)
        snap["index"] = key
        frames.append(snap)
        if verbose:
            print(f"{key}: {len(current)} current, {len(changes)} changes, "
                  f"{snap['ticker'].nunique()} distinct members over history")
    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(subset=["date", "ticker"]).sort_values(["date", "ticker"])
    full.to_csv(out, index=False)
    if verbose:
        allmembers = set(full["ticker"])
        print(f"Saved {out}: {len(full)} rows, {len(allmembers)} distinct tickers "
              f"across {full['date'].nunique()} monthly snapshots "
              f"({full['date'].min().date()} → {full['date'].max().date()})")
    return full


def load_membership(path="membership.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def universe_on(date, membership: pd.DataFrame) -> list[str]:
    """The tradeable universe as of `date` — nearest snapshot at or before it."""
    date = pd.Timestamp(date)
    avail = membership[membership["date"] <= date]
    if avail.empty:
        return []
    snap = avail["date"].max()
    return sorted(membership.loc[membership["date"] == snap, "ticker"].unique())


def delisted_members(membership: pd.DataFrame, current_universe) -> set[str]:
    """Historical members that are NOT in today's tradeable set — the names whose
    prices we most need and yfinance most likely lacks."""
    return set(membership["ticker"]) - set(current_universe)


def restrict_to_members(window: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not in the index as of the window's last date — enforces
    point-in-time SELECTION (you can't pick a name before it joined the index)."""
    members = set(universe_on(window.index[-1], membership))
    keep = [c for c in window.columns if c in members]
    return window[keep]


def pit_score(score_fn, membership: pd.DataFrame):
    """Wrap a score_fn so it only scores names that were index members at the
    rebalance date. Use as the backtester's score_fn for a PIT-honest universe."""
    def _f(window):
        w = restrict_to_members(window, membership)
        if w.shape[1] == 0:
            return pd.Series(dtype=float)
        return score_fn(w)
    return _f


def price_coverage(membership: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """How much of the point-in-time universe do we actually have prices for?"""
    members = set(membership["ticker"])
    have = members & set(prices.columns)
    missing = members - have
    return {"members": len(members), "have_prices": len(have),
            "missing_prices": len(missing),
            "coverage": len(have) / len(members) if members else float("nan"),
            "missing_sample": sorted(missing)[:20]}


if __name__ == "__main__":
    build_membership()
