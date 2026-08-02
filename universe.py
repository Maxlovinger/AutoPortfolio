"""
universe.py — build the tradeable stock universe (Phase 1: US small/mid, generous).

Pipeline:
  1. SOURCE   : pull S&P 500 + 400 + 600 constituents from Wikipedia (~1,500 names,
                the full S&P Composite 1500 — large+mid+small, generous by design).
  2. CLEAN    : normalize tickers for yfinance (BRK.B -> BRK-B), dedupe.
  3. ELIGIBLE : keep only tradeable names via mechanical filters
                (price floor, liquidity/ADV, price history, data completeness).
  4. SAVE     : write universe_candidates.csv (all) and universe.csv +
                universe_tickers.txt (the eligible, tradeable set).

SURVIVORSHIP CAVEAT: these are *today's* constituents, so any historical backtest
on them is optimistically biased. Use for directional research + FORWARD paper
trading (paper_trader.py), which is bias-free. Point-in-time membership needs a
paid vendor (Norgate/Sharadar/CRSP).
"""
from __future__ import annotations
import io
import urllib.request
import numpy as np
import pandas as pd
import yfinance as yf

SP_PAGES = {
    "SP500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "large"),
    "SP400": ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "mid"),
    "SP600": ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "small"),
}


def clean_ticker(t: str) -> str:
    """yfinance uses '-' for share classes (BRK.B -> BRK-B) and no dots."""
    return str(t).strip().upper().replace(".", "-")


def _read_wiki(url: str) -> pd.DataFrame:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "ignore")
    tables = pd.read_html(io.StringIO(html))
    for t in tables:
        cols = {str(c).lower(): c for c in t.columns}
        symcol = next((cols[k] for k in cols if "symbol" in k or "ticker" in k), None)
        if symcol is None:
            continue
        namecol = next((cols[k] for k in cols if "security" in k or "company" in k
                        or "name" in k), None)
        seccol = next((cols[k] for k in cols if "sector" in k or "industry" in k), None)
        out = pd.DataFrame({"ticker": t[symcol].map(clean_ticker)})
        out["name"] = t[namecol].astype(str) if namecol is not None else ""
        out["sector"] = t[seccol].astype(str) if seccol is not None else ""
        return out
    raise ValueError(f"No ticker table found at {url}")


def fetch_candidates() -> pd.DataFrame:
    """All S&P 1500 constituents with a size tier, deduped."""
    frames = []
    for key, (url, tier) in SP_PAGES.items():
        df = _read_wiki(url)
        df["tier"] = tier
        df["index"] = key
        frames.append(df)
    allc = pd.concat(frames, ignore_index=True)
    allc = allc.dropna(subset=["ticker"])
    allc = allc[allc["ticker"].str.len() > 0]
    allc = allc.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    return allc


def download_price_volume(tickers, period="5y", chunk=100):
    """Batched adjusted close + volume for many tickers. Returns (close, volume)."""
    closes, vols = [], []
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        raw = yf.download(part, period=period, auto_adjust=True,
                          progress=False, group_by="column", threads=True)
        if raw is None or len(raw) == 0:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            c = raw["Close"] if "Close" in raw.columns.levels[0] else pd.DataFrame()
            v = raw["Volume"] if "Volume" in raw.columns.levels[0] else pd.DataFrame()
        else:  # single ticker fell through
            c = raw[["Close"]].rename(columns={"Close": part[0]})
            v = raw[["Volume"]].rename(columns={"Volume": part[0]})
        closes.append(c)
        vols.append(v)
    close = pd.concat(closes, axis=1) if closes else pd.DataFrame()
    volume = pd.concat(vols, axis=1) if vols else pd.DataFrame()
    return close.sort_index(), volume.reindex(close.index)


def apply_eligibility(close: pd.DataFrame, volume: pd.DataFrame, *,
                      min_price=3.0, min_adv=1_000_000, min_days=504,
                      max_missing=0.10) -> pd.DataFrame:
    """
    Return a metrics DataFrame (index=ticker) with an `eligible` flag.
      min_price : last price floor (no penny stocks)
      min_adv   : median daily DOLLAR volume over last ~63 days (liquidity/capacity)
      min_days  : minimum non-NaN price observations (factors need history)
      max_missing: max fraction of NaNs allowed in the recent window
    """
    dollar_vol = close * volume
    rows = {}
    for t in close.columns:
        px = close[t].dropna()
        if len(px) == 0:
            continue
        last = float(px.iloc[-1])
        n_days = int(px.shape[0])
        adv = float(dollar_vol[t].tail(63).median()) if t in dollar_vol else np.nan
        recent = close[t].tail(min_days)
        missing = recent.isna().mean() if len(recent) else 1.0
        eligible = (last >= min_price and n_days >= min_days
                    and np.isfinite(adv) and adv >= min_adv
                    and missing <= max_missing)
        rows[t] = {"last_price": last, "adv_usd": adv, "n_days": n_days,
                   "missing": missing, "eligible": bool(eligible)}
    return pd.DataFrame(rows).T


def build_universe(top_n=2000, period="5y", min_price=3.0, min_adv=1_000_000,
                   min_days=504, out_prefix="universe", verbose=True):
    """Full pipeline. Saves candidate + eligible files. Returns eligible ticker list."""
    cand = fetch_candidates()
    cand.to_csv(f"{out_prefix}_candidates.csv", index=False)
    if verbose:
        print(f"Candidates: {len(cand)} tickers "
              f"({(cand.tier=='small').sum()} small, {(cand.tier=='mid').sum()} mid, "
              f"{(cand.tier=='large').sum()} large)")

    tickers = cand["ticker"].tolist()
    close, volume = download_price_volume(tickers, period=period)
    if verbose:
        print(f"Downloaded price history for {close.shape[1]} tickers")

    metrics = apply_eligibility(close, volume, min_price=min_price,
                                min_adv=min_adv, min_days=min_days)
    elig = metrics[metrics["eligible"]].copy()
    elig = elig.merge(cand.set_index("ticker")[["name", "sector", "tier"]],
                      left_index=True, right_index=True, how="left")
    # rank by liquidity, keep the most tradeable top_n (generous cap)
    elig = elig.sort_values("adv_usd", ascending=False).head(top_n)

    elig.to_csv(f"{out_prefix}.csv")
    with open(f"{out_prefix}_tickers.txt", "w") as f:
        f.write("\n".join(elig.index.tolist()))
    if verbose:
        print(f"Eligible & tradeable: {len(elig)} tickers "
              f"(saved {out_prefix}.csv / {out_prefix}_tickers.txt)")
        print(f"  tiers: {elig['tier'].value_counts().to_dict()}")
    return elig.index.tolist()


def load_universe(path="universe_tickers.txt") -> list[str]:
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


if __name__ == "__main__":
    build_universe()
