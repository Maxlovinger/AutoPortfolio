"""
databento_curve.py — pull the TRUE front/second futures curve from Databento
(GLBX.MDP3, CME group) to build REAL commodity carry, replacing the FRED
spot-basis proxy.

Uses Databento CONTINUOUS symbology: `{ROOT}.n.0` = front month, `{ROOT}.n.1` =
second month (`.n.` = roll on open interest). Front-vs-second is exactly the
carry slope, in the SAME units (no proxy / no unit-offset problem), back to
2010-06 (GLBX start) — ~16 years incl. the 2014-2020 commodity bear the 6-year
proxy was missing.

Cost: daily bars (`ohlcv-1d`) for 26 symbols over 16 years is a few MB. The script
calls metadata.get_cost() FIRST and aborts if it somehow exceeds a safety cap.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from utils import MONTH_END

RAW_CACHE = os.path.join(os.path.dirname(__file__), ".commodity_cache",
                        "databento_curve.pkl")
DATASET = "GLBX.MDP3"          # CME Globex: CME + CBOT + NYMEX + COMEX
SCHEMA = "ohlcv-1d"

# our market name -> CME Globex root symbol (13 CME-group markets; softs on ICE)
ROOTS = {
    "Oil": "CL", "NatGas": "NG", "Gasoline": "RB", "HeatingOil": "HO",
    "Gold": "GC", "Silver": "SI", "Copper": "HG", "Platinum": "PL",
    "Corn": "ZC", "Wheat": "ZW", "Soybeans": "ZS", "SoyOil": "ZL",
    "LiveCattle": "LE",
}
DEPTHS = (0, 1)               # 0 = front, 1 = second month


def symbols() -> list[str]:
    return [f"{r}.n.{d}" for r in ROOTS.values() for d in DEPTHS]


def _client():
    from fx.data import load_dotenv
    load_dotenv()
    import databento as db
    key = os.getenv("DATABENTO_API")
    if not key:
        raise RuntimeError("DATABENTO_API not found in .env")
    return db.Historical(key)


def _end(end):
    # continuous symbology fails to resolve with end=None -> use an explicit date
    return end or pd.Timestamp.today().strftime("%Y-%m-%d")


def estimate_cost(start="2010-06-06", end=None) -> float:
    c = _client()
    return c.metadata.get_cost(dataset=DATASET, schema=SCHEMA,
                               stype_in="continuous", symbols=symbols(),
                               start=start, end=_end(end))


def download(start="2010-06-06", end=None, max_cost=10.0,
             use_cache=True) -> pd.DataFrame:
    """get_cost -> confirm under cap -> pull -> cache the raw long df."""
    if use_cache and os.path.exists(RAW_CACHE):
        return pd.read_pickle(RAW_CACHE)
    end = _end(end)
    c = _client()
    cost = c.metadata.get_cost(dataset=DATASET, schema=SCHEMA,
                               stype_in="continuous", symbols=symbols(),
                               start=start, end=end)
    print(f"Databento estimated cost: ${cost:.4f}", flush=True)
    if cost > max_cost:
        raise RuntimeError(f"cost ${cost:.2f} exceeds cap ${max_cost}; aborting")
    data = c.timeseries.get_range(dataset=DATASET, schema=SCHEMA,
                                  stype_in="continuous", symbols=symbols(),
                                  start=start, end=end)
    df = data.to_df()
    os.makedirs(os.path.dirname(RAW_CACHE), exist_ok=True)
    df.to_pickle(RAW_CACHE)
    print(f"downloaded {len(df)} rows -> {RAW_CACHE}", flush=True)
    return df


# --- shape into front/second panels + carry --------------------------------
def curve_panels(raw: pd.DataFrame | None = None):
    """Raw long df -> (front_close, second_close) daily panels, columns=markets."""
    raw = raw if raw is not None else pd.read_pickle(RAW_CACHE)
    df = raw.reset_index()
    sym = df["symbol"].astype(str)
    df["root"] = sym.str.split(".").str[0]
    df["depth"] = sym.str.rsplit(".", n=1).str[-1].astype(int)
    root2name = {v: k for k, v in ROOTS.items()}
    df["market"] = df["root"].map(root2name)
    tcol = "ts_event" if "ts_event" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[tcol], utc=True).dt.tz_localize(None).dt.normalize()
    front = df[df.depth == 0].pivot_table(index="date", columns="market",
                                          values="close", aggfunc="last")
    second = df[df.depth == 1].pivot_table(index="date", columns="market",
                                           values="close", aggfunc="last")
    return front, second


def carry_monthly(front: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly TRUE carry = annualized log slope of front vs second contract:
        carry = (log(front) - log(second)) * 12
    Same units (real curve), so positive = backwardation = long candidate. The
    x12 assumes ~1-month spacing (rough for quarterly-cycle markets; a per-market
    z-score downstream removes constant scaling)."""
    f = front.resample(MONTH_END).last()
    s = second.resample(MONTH_END).last()
    cols = [c for c in f.columns if c in s.columns]
    return (np.log(f[cols]) - np.log(s[cols])) * 12


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cost":
        print(f"estimated cost: ${estimate_cost():.4f}")
        raise SystemExit
    raw = download()
    print(f"\nraw columns: {list(raw.columns)}")
    front, second = curve_panels(raw)
    carry = carry_monthly(front, second)
    print(f"\nfront panel: {front.shape[1]} markets, "
          f"{front.index.min().date()}..{front.index.max().date()}")
    print("coverage (months of carry) per market:")
    for name, n in carry.notna().sum().sort_values().items():
        print(f"  {name:<12}{n:>4}")
    # sanity: latest carry (should be signed, plausible magnitudes)
    print("\nlatest monthly carry (annualized, + = backwardation):")
    print(carry.dropna(how="all").iloc[-1].round(3).to_string())
