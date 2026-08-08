"""
equity_sentiment.py — does news sentiment improve the equity book? (honest test)

Ties the GDELT news-tone signal into the equity strategy and backtests whether
it beats what we already run. The disciplined design mirrors the FX composite
and allocation bake-offs: hold EVERYTHING ELSE fixed (same liquid basket, same
quarterly cadence, same 15% vol-target overlay) and only change the WEIGHTING —
equal-weight (current) vs a sentiment TILT. So any difference is the sentiment
signal, not a confound.

Why this is finally testable: GDELT tone is timestamped, so it's point-in-time
safe (unlike yfinance headlines, which are current-only and were EXCLUDED from
the backtester for exactly that reason). The tilt at each rebalance uses only
tone from months that ENDED strictly before the decision date.

Data note: GDELT rate-limits/blocks datacenter IPs — build the tone panel on a
normal machine (it caches to disk); the backtest itself is then offline.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from backtester import walk_forward, weight_equal, vol_target, performance

CACHE = os.path.join(os.path.dirname(__file__), "tone_equity.pkl")


# --- ticker -> company name (GDELT matches names, not tickers) --------------
def ticker_names(tickers, universe_csv="universe.csv") -> dict:
    """Map tickers to company names from universe.csv for GDELT queries."""
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), universe_csv),
                     index_col=0)
    out = {}
    for t in tickers:
        if t in df.index and isinstance(df.loc[t, "name"], str):
            out[t] = df.loc[t, "name"]
    return out


def build_tone_panel(tickers, timespan="72m", cache=CACHE,
                     universe_csv="universe.csv") -> pd.DataFrame:
    """Monthly GDELT entity-tone panel (one column per ticker). Cached to disk;
    fetch happens once, from a non-datacenter IP. Reruns load the cache."""
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    from news_gdelt import entity_tone
    names = ticker_names(tickers, universe_csv)
    cols = {}
    n = len(names)
    print(f"Fetching GDELT tone for {n} names. Each query caches to .gdelt_cache/, "
          f"so a Ctrl-C + rerun resumes instantly for names already fetched.",
          flush=True)
    for i, (t, name) in enumerate(names.items(), 1):
        try:
            s = entity_tone(name, timespan=timespan)   # throttle is inside _fetch
            got = f"{len(s)} pts"
        except Exception as e:
            s, got = None, f"FAILED ({str(e)[:30]})"
        print(f"  [{i:>2}/{n}] {t:<6} {name[:26]:<26} -> {got}", flush=True)
        if s is not None and not s.empty:
            cols[t] = s
    panel = pd.DataFrame(cols)
    if not panel.empty:
        panel.to_pickle(cache)           # write the FINAL panel only when complete
    print(f"Done: usable tone for {panel.shape[1]}/{n} names -> {cache}", flush=True)
    return panel


# --- the sentiment tilt (no look-ahead) ------------------------------------
def _naive_tone(tone_panel):
    """Ensure the tone index is tz-naive so it compares against tz-naive price
    dates (GDELT stamps are UTC; older cached panels may still carry tz)."""
    if tone_panel is not None and getattr(tone_panel.index, "tz", None) is not None:
        tone_panel = tone_panel.copy()
        tone_panel.index = tone_panel.index.tz_localize(None)
    return tone_panel


def _cap_waterfill(w, cap, iters=50):
    """Enforce a per-name cap while keeping sum=1 and preserving the ordering of
    the uncapped names: clip over-cap weights to `cap` and redistribute the
    excess proportionally among the still-uncapped names, repeated to convergence.
    (Plain clip-then-renormalize breaks the cap and flattens distinct weights.)"""
    w = w.copy()
    if cap is None or cap >= 1.0 or len(w) * cap < 1.0:
        return w                                     # no cap, or infeasible
    for _ in range(iters):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        pool = w[under].sum()
        if pool <= 0:
            break
        w[under] += excess * w[under] / pool
    return w.clip(upper=cap)


def sentiment_tilt_weight(window, picks, tone_panel=None, lam=0.5, cap=0.15,
                          zclip=2.0):
    """Equal-weight base, tilted toward higher-tone names. Uses only tone from
    months that ended BEFORE the decision date (window's last day) — no leak.
    Long-only, per-name cap enforced by water-filling. Falls back to equal weight
    when tone is missing or has no cross-sectional dispersion."""
    n = len(picks)
    eq = pd.Series(1.0 / n, index=picks)
    if tone_panel is None:
        return eq
    t = window.index[-1]
    past = tone_panel.loc[tone_panel.index < t]          # strictly-earlier months
    if past.empty:
        return eq
    tone_now = past.ffill().iloc[-1].reindex(picks)
    if tone_now.notna().sum() < 2 or tone_now.std(ddof=0) == 0:
        return eq
    z = ((tone_now - tone_now.mean()) / tone_now.std(ddof=0)).clip(-zclip, zclip)
    raw = ((1.0 / n) * (1.0 + lam * z.fillna(0.0))).clip(lower=0.0)
    if raw.sum() == 0:
        return eq
    return _cap_waterfill(raw / raw.sum(), cap)


# --- comparison harness ----------------------------------------------------
def _all_names_score(window):
    """Constant score -> selection is the whole (fixed liquid) basket; the test
    is about weighting, not selection."""
    return pd.Series(1.0, index=window.columns)


def run_compare(prices, tone_panel, *, lookback=252, rebalance=63,
                target_vol=0.15, cost_bps=10.0, lam=0.5, cap=0.15,
                train_end="2022-12-31"):
    """Baseline (equal-weight) vs sentiment-tilt on the same fixed basket, both
    with the 15% vol-target overlay. Returns full-sample and held-out (>train_end)
    metrics for each — the honest bar is the TEST slice."""
    from functools import partial
    n = prices.shape[1]
    tone_panel = _naive_tone(tone_panel)

    base = walk_forward(prices, _all_names_score, weight_equal,
                        top_n=n, lookback=lookback, rebalance=rebalance,
                        cost_bps=cost_bps)
    tilt = walk_forward(prices, _all_names_score,
                        partial(sentiment_tilt_weight, tone_panel=tone_panel,
                                lam=lam, cap=cap),
                        top_n=n, lookback=lookback, rebalance=rebalance,
                        cost_bps=cost_bps)

    out = {}
    cut = pd.Timestamp(train_end)
    for name, res in (("equal-weight", base), ("sentiment-tilt", tilt)):
        r = vol_target(res["returns"], target=target_vol)          # deployed overlay
        out[name] = {"full": performance(r),
                     "test": performance(r[r.index > cut])}
    return out


# --- PIT (survivorship-free) comparison ------------------------------------
def _pit_setup(prices_path, basket_n, max_per_sector):
    """The exact PIT selection from final_strategy: score by liquidity gated to
    point-in-time membership, then pick the most-liquid names sector-capped."""
    import historical_membership as hm
    from sector_select import load_sectors, select_sector_capped
    from costs import load_adv
    from allocation_bakeoff import make_score_liquid

    prices = pd.read_pickle(prices_path).dropna(how="all", axis=1).sort_index()
    membership = hm.load_membership()
    sectors = load_sectors("universe.csv")
    adv = load_adv("universe.csv")
    score_fn = hm.pit_score(make_score_liquid(adv), membership)
    select_fn = lambda s: select_sector_capped(
        s, sectors, top_n=basket_n, max_per_sector=max_per_sector)
    return prices, score_fn, select_fn, adv


def names_ever_held(prices_path="prices_pit.pkl", basket_n=30, max_per_sector=5,
                    lookback=252, rebalance=63, capital=5_000_000.0) -> list:
    """Union of every name the PIT strategy ever holds over the backtest — the
    exact set we must fetch tone for (today's 30 is NOT enough; 5y ago the book
    was different)."""
    from allocation_bakeoff import make_weight
    prices, score_fn, select_fn, adv = _pit_setup(
        prices_path, basket_n, max_per_sector)
    res = walk_forward(prices, score_fn, make_weight("equal"), select_fn=select_fn,
                       lookback=lookback, rebalance=rebalance, adv=adv,
                       capital=capital)
    W = res["weights"]
    return sorted([c for c in W.columns if W[c].abs().sum() > 0])


def run_compare_pit(tone_panel, *, prices_path="prices_pit.pkl", basket_n=30,
                    max_per_sector=5, lookback=252, rebalance=63,
                    target_vol=0.15, capital=5_000_000.0, lam=0.5, cap=0.15,
                    train_end="2022-12-31"):
    """Survivorship-FREE comparison: the deployed PIT liquid-30 book, equal-weight
    vs sentiment-tilted. At each past rebalance the picks are whatever was liquid
    THEN, and the tilt uses those names' tone AT THAT TIME. Realistic per-name
    costs (adv) + 15% vol-target, held-out at train_end."""
    from functools import partial
    from allocation_bakeoff import make_weight
    tone_panel = _naive_tone(tone_panel)
    prices, score_fn, select_fn, adv = _pit_setup(
        prices_path, basket_n, max_per_sector)

    base = walk_forward(prices, score_fn, make_weight("equal"), select_fn=select_fn,
                        lookback=lookback, rebalance=rebalance, adv=adv,
                        capital=capital)
    tilt = walk_forward(prices, score_fn,
                        partial(sentiment_tilt_weight, tone_panel=tone_panel,
                                lam=lam, cap=cap),
                        select_fn=select_fn, lookback=lookback,
                        rebalance=rebalance, adv=adv, capital=capital)

    out = {}
    cut = pd.Timestamp(train_end)
    for name, res in (("equal-weight PIT", base), ("sentiment-tilt PIT", tilt)):
        r = vol_target(res["returns"], target=target_vol)
        out[name] = {"full": performance(r),
                     "test": performance(r[r.index > cut])}
    return out


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "pit"

    if mode == "pit":
        # SURVIVORSHIP-FREE: fetch tone for every name ever held, then compare.
        names = names_ever_held()
        print(f"PIT strategy held {len(names)} distinct names over the backtest; "
              f"fetching tone for all of them...")
        tone = build_tone_panel(names, cache="tone_equity_pit.pkl")
        if tone.empty:
            print("No GDELT tone (datacenter IP 429-blocked?). Run on a normal "
                  "machine to build tone_equity_pit.pkl, then rerun.")
            sys.exit(0)
        res = run_compare_pit(tone)
        print(f"\n{'book':20}{'FULL Sharpe':>13}{'FULL MaxDD':>12}"
              f"{'TEST Sharpe':>13}{'TEST MaxDD':>12}")
        for name, m in res.items():
            f, t = m["full"], m["test"]
            print(f"{name:20}{f['sharpe']:>13.3f}{f['max_dd']:>12.3f}"
                  f"{t['sharpe']:>13.3f}{t['max_dd']:>12.3f}")
        sys.exit(0)

    # mode == "fixed": quick survivorship-BIASED look on today's liquid-30
    prices = pd.read_pickle("prices_universe.pkl")
    # fixed liquid basket = the N most-liquid names by ADV (survivorship-caveated,
    # matches the allocation-bakeoff precedent); swap in prices_pit.pkl+membership
    # for a fully PIT run.
    uni = pd.read_csv("universe.csv", index_col=0)
    liquid = (uni[uni["eligible"]].sort_values("adv_usd", ascending=False)
              .head(30).index)
    liquid = [t for t in liquid if t in prices.columns]
    prices = prices[liquid].dropna(how="all")

    tone = build_tone_panel(liquid)
    if tone.empty:
        print("No GDELT tone fetched (datacenter IP is 429-blocked?). "
              "Run this on a normal machine to build tone_equity.pkl, then rerun.")
        sys.exit(0)

    res = run_compare(prices, tone)
    print(f"{'book':16}{'FULL Sharpe/MaxDD':>26}{'TEST Sharpe/MaxDD':>26}")
    for name, m in res.items():
        f, t = m["full"], m["test"]
        print(f"{name:16}{f['sharpe']:>14.3f}{f['max_dd']:>12.3f}"
              f"{t['sharpe']:>14.3f}{t['max_dd']:>12.3f}")
