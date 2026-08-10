"""
commodity_futures.py — Tier-2 futures data layer (build-order step 1).

Ingests real futures data from Interactive Brokers (TWS, via ibapi) and turns it
into the two objects the strategy needs, mirroring fx/data.py:

  * CONTINUOUS back-adjusted daily series  -> trend signal + return realization
  * the FRONT-vs-NEXT curve slope           -> carry signal (roll yield)

Both are built from the SAME ingested contracts with ONE roll rule, so carry and
trend stay consistent (the roll yield shows up once in carry and never as a fake
jump in trend — see the roll-convention discussion).

Two-phase ingest (IBKR historical-data pacing is the constraint):
  Phase 1  ingest_continuous()  — one CONTFUT request per market (fast, ~15 reqs)
           -> enables the TREND backtest immediately.
  Phase 2  ingest_curve()       — pull the individual contract chain per market and
           reconstruct the historical term structure (front/next) -> CARRY. Slower
           (many requests, paced + cached per contract), run in the background.

The pure logic (roll/back-adjust, carry slope, front/next selection) is separated
out and unit-tested; the ibapi fetches are integration (run against live TWS).
"""
from __future__ import annotations
import os
import time
import threading
import numpy as np
import pandas as pd

from utils import MONTH_END

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".commodity_cache", "fut")
PANEL_CACHE = os.path.join(os.path.dirname(__file__), "commodity_futures.pkl")

# --- universe: liquid futures across sectors (IBKR symbol/exchange/currency) --
# mult = contract multiplier (for notional; not needed for returns/carry).
FUTURES = {
    # energy (NYMEX)
    "Oil":       {"symbol": "CL", "exchange": "NYMEX", "ccy": "USD", "sector": "energy", "mult": 1000},
    "NatGas":    {"symbol": "NG", "exchange": "NYMEX", "ccy": "USD", "sector": "energy", "mult": 10000},
    "Gasoline":  {"symbol": "RB", "exchange": "NYMEX", "ccy": "USD", "sector": "energy", "mult": 42000},
    "HeatingOil":{"symbol": "HO", "exchange": "NYMEX", "ccy": "USD", "sector": "energy", "mult": 42000},
    # metals (COMEX / NYMEX)
    "Gold":      {"symbol": "GC", "exchange": "COMEX", "ccy": "USD", "sector": "metals", "mult": 100},
    "Silver":    {"symbol": "SI", "exchange": "COMEX", "ccy": "USD", "sector": "metals", "mult": 5000},
    "Copper":    {"symbol": "HG", "exchange": "COMEX", "ccy": "USD", "sector": "metals", "mult": 25000},
    "Platinum":  {"symbol": "PL", "exchange": "NYMEX", "ccy": "USD", "sector": "metals", "mult": 50},
    # grains / oilseeds (CBOT)
    "Corn":      {"symbol": "ZC", "exchange": "CBOT",  "ccy": "USD", "sector": "grains", "mult": 5000},
    "Wheat":     {"symbol": "ZW", "exchange": "CBOT",  "ccy": "USD", "sector": "grains", "mult": 5000},
    "Soybeans":  {"symbol": "ZS", "exchange": "CBOT",  "ccy": "USD", "sector": "grains", "mult": 5000},
    "SoyOil":    {"symbol": "ZL", "exchange": "CBOT",  "ccy": "USD", "sector": "grains", "mult": 60000},
    # softs (ICE / NYBOT)
    "Sugar":     {"symbol": "SB", "exchange": "NYBOT", "ccy": "USD", "sector": "softs",  "mult": 112000},
    "Coffee":    {"symbol": "KC", "exchange": "NYBOT", "ccy": "USD", "sector": "softs",  "mult": 37500},
    "Cotton":    {"symbol": "CT", "exchange": "NYBOT", "ccy": "USD", "sector": "softs",  "mult": 50000},
    # livestock (CME)
    "LiveCattle":{"symbol": "LE", "exchange": "CME",   "ccy": "USD", "sector": "meats",  "mult": 40000},
}


# ===========================================================================
# PURE LOGIC (unit-tested; no network)
# ===========================================================================
def annualized_carry(front_px: float, next_px: float, days_between: int) -> float:
    """
    Annualized roll yield from the two nearest contracts:
        carry = (front - next) / next * (365 / days_between)
    front > next (backwardation) -> positive carry (a long candidate).
    """
    if next_px <= 0 or days_between <= 0:
        return np.nan
    return (front_px - next_px) / next_px * (365.0 / days_between)


def pick_front_next(expiries: list[str], on_date: pd.Timestamp,
                    min_days=5):
    """
    Given sorted contract expiries (YYYYMMDD strings) and a date, return
    (front_expiry, next_expiry): the nearest contract expiring at least
    `min_days` ahead (front) and the one after it (next). None if unavailable.
    `min_days` avoids holding into the delivery/first-notice window.
    """
    exps = sorted(pd.Timestamp(e) for e in expiries)
    live = [e for e in exps if (e - on_date).days >= min_days]
    if len(live) < 2:
        return (None, None)
    return (live[0].strftime("%Y%m%d"), live[1].strftime("%Y%m%d"))


def back_adjust(contract_frames: dict, roll_dates: dict) -> pd.Series:
    """
    Stitch front contracts into ONE back-adjusted continuous close series, using
    PROPORTIONAL (ratio) adjustment — the right choice because the whole pipeline
    trades on percentage returns (vol-target, Sharpe, trend). Ratio adjustment
    preserves each contract's within-contract % returns EXACTLY and removes the
    roll gap; additive adjustment would preserve $ changes but distort % returns.

    contract_frames : {expiry -> DataFrame with a 'close' column, datetime index}
    roll_dates      : {expiry -> the date we roll OUT of that contract INTO the
                       next}, so contract `e` is "held" up to roll_dates[e].

    On each roll we take the ratio (new_contract / old_contract) at the roll date
    and MULTIPLY all earlier prices by it, so the continuous line has no artificial
    jump. Absolute level is synthetic; returns are what matter.
    """
    exps = sorted(contract_frames, key=lambda e: pd.Timestamp(e))
    # assemble the raw held segments first (each contract up to its roll date)
    segs = []
    prev_roll = pd.Timestamp.min
    for e in exps:
        df = contract_frames[e].sort_index()
        roll = pd.Timestamp(roll_dates.get(e, df.index.max()))
        seg = df.loc[(df.index > prev_roll) & (df.index <= roll), "close"]
        if len(seg):
            segs.append((e, seg, roll))
        prev_roll = roll
    if not segs:
        return pd.Series(dtype=float)

    # walk backwards, accumulating the multiplicative factor at each roll boundary
    cont = {}
    factor = 1.0
    for i in range(len(segs) - 1, -1, -1):
        e, seg, roll = segs[i]
        for dt, px in seg.items():
            cont[dt] = px * factor
        if i > 0:                                  # gap at the boundary with prior seg
            e_old = segs[i - 1][0]
            old_df = contract_frames[e_old].sort_index()
            new_df = contract_frames[e].sort_index()
            boundary = segs[i - 1][2]              # prior seg's roll date
            # ratio of NEW to OLD contract at the roll -> the multiplicative gap
            if (boundary in old_df.index and boundary in new_df.index
                    and old_df.loc[boundary, "close"]):
                factor *= (new_df.loc[boundary, "close"]
                           / old_df.loc[boundary, "close"])
    return pd.Series(cont).sort_index()


def build_carry_series(front_frames: pd.DataFrame) -> pd.Series:
    """
    front_frames : DataFrame indexed by date with columns
        ['front_px','next_px','days_between'] (the term structure over time)
    -> daily annualized carry series.
    """
    f = front_frames.dropna(subset=["front_px", "next_px", "days_between"])
    return pd.Series(
        [annualized_carry(r.front_px, r.next_px, r.days_between)
         for r in f.itertuples()], index=f.index, name="carry")


# ===========================================================================
# IBKR CLIENT (integration; needs TWS)
# ===========================================================================
def _make_client():
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper

    class FuturesClient(EWrapper, EClient):
        def __init__(self):
            EClient.__init__(self, self)
            self._bars = {}
            self._chain = {}
            self._done = set()
            self._cd_done = set()
            self.errors = []

        def error(self, reqId, code, msg, *a):
            if code not in (2104, 2106, 2158, 2107, 2119):   # benign data-farm notices
                self.errors.append((reqId, code, str(msg)[:80]))

        def historicalData(self, reqId, bar):
            self._bars.setdefault(reqId, []).append(
                (bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume))

        def historicalDataEnd(self, reqId, s, e):
            self._done.add(reqId)

        def contractDetails(self, reqId, cd):
            self._chain.setdefault(reqId, []).append(
                cd.contract.lastTradeDateOrContractMonth)

        def contractDetailsEnd(self, reqId):
            self._cd_done.add(reqId)

    return FuturesClient()


def _connect(host="127.0.0.1", port=7497, client_id=51):
    app = _make_client()
    app.connect(host, port, client_id)
    threading.Thread(target=app.run, daemon=True).start()
    time.sleep(1.5)
    return app


def _contract(cfg, secType, expiry=None):
    from ibapi.contract import Contract
    c = Contract()
    c.symbol = cfg["symbol"]; c.secType = secType
    c.exchange = cfg["exchange"]; c.currency = cfg["ccy"]
    if expiry:
        c.lastTradeDateOrContractMonth = expiry
    return c


def _wait(app, cond, timeout):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        time.sleep(0.3)


def _bars_to_df(rows):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    return df.dropna(subset=["date"]).set_index("date").sort_index()


_REQ = [0]
def _next_req():
    _REQ[0] += 1
    return _REQ[0]


def fetch_continuous(app, market, duration="15 Y", timeout=30) -> pd.DataFrame:
    """CONTFUT continuous daily bars for one market (fast; one request)."""
    cfg = FUTURES[market]
    rid = _next_req()
    c = _contract(cfg, "CONTFUT")
    app.reqHistoricalData(rid, c, "", duration, "1 day", "TRADES", 0, 1, False, [])
    _wait(app, lambda: rid in app._done, timeout)
    return _bars_to_df(app._bars.get(rid, []))


# ===========================================================================
# INGEST
# ===========================================================================
def ingest_continuous(markets=None, duration="15 Y", pace=2.0,
                      client_id=51) -> pd.DataFrame:
    """
    Phase 1: continuous daily close per market -> a cached price panel. Enables the
    trend backtest. Per-market try/except so one bad exchange doesn't sink the run.
    """
    markets = markets or list(FUTURES)
    app = _connect(client_id=client_id)
    cols = {}
    print(f"Ingesting continuous futures for {len(markets)} markets...", flush=True)
    for i, m in enumerate(markets, 1):
        try:
            df = fetch_continuous(app, m, duration=duration)
            note = (f"{len(df)} bars, {df.index.min().date()}..{df.index.max().date()}"
                    if len(df) else "EMPTY")
            if len(df):
                cols[m] = df["close"]
        except Exception as e:
            note = f"FAILED ({str(e)[:40]})"
        print(f"  [{i:>2}/{len(markets)}] {m:<11} -> {note}", flush=True)
        time.sleep(pace)                              # respect IBKR pacing
    app.disconnect()
    panel = pd.DataFrame(cols).sort_index()
    if len(panel):
        panel.to_pickle(PANEL_CACHE)
        print(f"cached continuous panel -> {PANEL_CACHE}")
    if app.errors:
        print("IBKR notices:", app.errors[:8])
    return panel


def load_continuous(monthly=True) -> pd.DataFrame:
    """Cached continuous panel; monthly=True -> month-end grid (trade grid)."""
    panel = pd.read_pickle(PANEL_CACHE)
    return panel.resample(MONTH_END).last() if monthly else panel


if __name__ == "__main__":
    import sys
    dur = sys.argv[1] if len(sys.argv) > 1 else "15 Y"
    panel = ingest_continuous(duration=dur)
    if len(panel):
        m = panel.resample(MONTH_END).last()
        print(f"\nMonthly continuous panel: {m.shape[1]} markets, "
              f"{m.index.min():%Y-%m}..{m.index.max():%Y-%m} ({len(m)} months)")
        print("coverage (months) per market:")
        for name, n in m.notna().sum().sort_values().items():
            print(f"  {name:<11}{n:>4}")
