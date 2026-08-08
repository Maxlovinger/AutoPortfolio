"""
earnings_signal.py — PEAD tilt: overweight recent positive earnings surprises.

The disciplined test of "does earnings news improve the book", built on the same
harness as the sentiment tilt (equity_sentiment.py) and judged the same way
(survivorship-free PIT, no look-ahead, vs a random-signal null).

Signal at rebalance date t, per held name: the SURPRISE(%) of its most recent
earnings announced in the drift window (t - window_days, t) — 0 if it hasn't
reported recently. Cross-sectionally z-scored, it tilts equal weight toward
recent beats (PEAD) via the same water-filling cap.

Optional refinement (user asked for surprise + FinBERT): pass a `tone_panel`
(GDELT/FinBERT earnings-headline tone) to BLEND with the numeric surprise. The
surprise is the documented driver; tone is the add-on.

Cadence note: PEAD plays out over ~40-60 days post-announcement, so this signal
wants a MONTHLY rebalance and a broad universe (small/mid-cap, where PEAD is
strongest). On the mega-cap liquid-30 quarterly book it's expected weak — that
comparison is the honest 'does it help what we deploy' check.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from backtester import walk_forward, weight_equal, vol_target, performance
from equity_sentiment import _cap_waterfill, _naive_tone


def recent_surprise(surprise_panel, picks, t, window_days=90) -> pd.Series:
    """Most-recent Surprise(%) per pick within (t-window, t); 0 if none. Strictly
    earlier than t -> no look-ahead."""
    lo = t - pd.Timedelta(days=window_days)
    mask = (surprise_panel.index < t) & (surprise_panel.index >= lo)
    sub = surprise_panel.loc[mask]
    out = {}
    for c in picks:
        s = sub[c].dropna() if c in sub.columns else pd.Series(dtype=float)
        out[c] = float(s.iloc[-1]) if len(s) else 0.0
    return pd.Series(out, index=picks)


def earnings_tilt_weight(window, picks, surprise_panel=None, window_days=90,
                         lam=0.5, cap=0.15, tone_panel=None, tone_w=0.5):
    """Equal-weight base tilted by recent earnings surprise (+ optional tone
    blend). Long-only, water-filling cap, no look-ahead. Falls back to equal
    weight when no recent earnings / no dispersion."""
    n = len(picks)
    eq = pd.Series(1.0 / n, index=picks)
    if surprise_panel is None:
        return eq
    t = window.index[-1]

    def _z(sig):
        if sig.abs().sum() == 0 or sig.std(ddof=0) == 0:
            return pd.Series(0.0, index=picks)
        return ((sig - sig.mean()) / sig.std(ddof=0)).clip(-2, 2).fillna(0.0)

    z = _z(recent_surprise(surprise_panel, picks, t, window_days))
    if tone_panel is not None:                       # blend in FinBERT tone
        past = tone_panel.loc[tone_panel.index < t]
        if not past.empty:
            tone_now = past.ffill().iloc[-1].reindex(picks)
            z = (1 - tone_w) * z + tone_w * _z(tone_now.fillna(0.0))
    if z.abs().sum() == 0:
        return eq
    raw = ((1.0 / n) * (1.0 + lam * z)).clip(lower=0.0)
    if raw.sum() == 0:
        return eq
    return _cap_waterfill(raw / raw.sum(), cap)


# --- comparison harness (fixed basket + PIT), reusing equity_sentiment ------
def _summ(res, target_vol, train_end):
    r = vol_target(res["returns"], target=target_vol)
    cut = pd.Timestamp(train_end)
    return {"full": performance(r), "test": performance(r[r.index > cut])}


def run_compare_fixed(prices, surprise_panel, *, lookback=252, rebalance=63,
                      target_vol=0.15, cost_bps=10.0, lam=0.5, cap=0.15,
                      window_days=90, tone_panel=None, train_end="2022-12-31"):
    """Earnings tilt vs equal-weight on a FIXED basket (survivorship-biased,
    quick look)."""
    from functools import partial
    from equity_sentiment import _all_names_score
    n = prices.shape[1]
    tone_panel = _naive_tone(tone_panel)
    base = walk_forward(prices, _all_names_score, weight_equal, top_n=n,
                        lookback=lookback, rebalance=rebalance, cost_bps=cost_bps)
    tilt = walk_forward(prices, _all_names_score,
                        partial(earnings_tilt_weight, surprise_panel=surprise_panel,
                                window_days=window_days, lam=lam, cap=cap,
                                tone_panel=tone_panel),
                        top_n=n, lookback=lookback, rebalance=rebalance,
                        cost_bps=cost_bps)
    return {"equal-weight": _summ(base, target_vol, train_end),
            "earnings-tilt": _summ(tilt, target_vol, train_end)}


def run_compare_pit(surprise_panel, *, prices_path="prices_pit.pkl", basket_n=30,
                    max_per_sector=5, lookback=252, rebalance=63, target_vol=0.15,
                    capital=5_000_000.0, lam=0.5, cap=0.15, window_days=90,
                    tone_panel=None, train_end="2022-12-31"):
    """Survivorship-FREE: the deployed PIT book, equal-weight vs earnings-tilt.
    `rebalance`=63 mirrors the deployed quarterly book; pass 21 for the
    monthly 'fair shot' that actually catches the drift window."""
    from functools import partial
    from equity_sentiment import _pit_setup
    from allocation_bakeoff import make_weight
    tone_panel = _naive_tone(tone_panel)
    prices, score_fn, select_fn, adv = _pit_setup(
        prices_path, basket_n, max_per_sector)
    base = walk_forward(prices, score_fn, make_weight("equal"), select_fn=select_fn,
                        lookback=lookback, rebalance=rebalance, adv=adv,
                        capital=capital)
    tilt = walk_forward(prices, score_fn,
                        partial(earnings_tilt_weight, surprise_panel=surprise_panel,
                                window_days=window_days, lam=lam, cap=cap,
                                tone_panel=tone_panel),
                        select_fn=select_fn, lookback=lookback, rebalance=rebalance,
                        adv=adv, capital=capital)
    return {"equal-weight PIT": _summ(base, target_vol, train_end),
            "earnings-tilt PIT": _summ(tilt, target_vol, train_end)}


def make_earnings_score(surprise_panel, membership, window_days=90):
    """score_fn for walk_forward: rank PIT members by their most-recent earnings
    surprise (the PEAD long signal), gated to point-in-time membership."""
    import historical_membership as hm

    def score(window):
        t = window.index[-1]
        cols = [c for c in window.columns if c in surprise_panel.columns]
        sig = recent_surprise(surprise_panel, cols, t, window_days)
        members = set(hm.universe_on(t, membership))
        sig = sig[[c for c in sig.index if c in members]]
        return sig[sig != 0.0]                       # only names that recently reported
    return score


def run_pead_selection(surprise_panel, tickers, *, prices_path="prices_pit.pkl",
                       top_n=40, max_per_sector=6, lookback=252, rebalance=21,
                       target_vol=0.15, capital=5_000_000.0, window_days=90,
                       cost_bps=10.0, train_end="2022-12-31"):
    """FAIR SHOT: from a BROAD PIT universe, each month LONG the top-N biggest
    recent earnings beats (sector-capped, equal-weight, vol-target). Compared vs
    equal-weighting the same broad universe. This is PEAD's real habitat."""
    import historical_membership as hm
    from sector_select import load_sectors, select_sector_capped
    from allocation_bakeoff import make_score_liquid, make_weight
    from costs import load_adv

    prices = pd.read_pickle(prices_path).dropna(how="all", axis=1).sort_index()
    prices = prices[[t for t in tickers if t in prices.columns]]
    membership = hm.load_membership()
    sectors = load_sectors("universe.csv")
    adv = load_adv("universe.csv")

    sel = lambda s: select_sector_capped(s, sectors, top_n=top_n,
                                         max_per_sector=max_per_sector)
    pead = walk_forward(prices, make_earnings_score(surprise_panel, membership,
                                                    window_days),
                        make_weight("equal"), select_fn=sel, lookback=lookback,
                        rebalance=rebalance, adv=adv, capital=capital)
    # benchmark: equal-weight the broad liquid universe (PIT), same cadence
    bench = walk_forward(prices, hm.pit_score(make_score_liquid(adv), membership),
                         make_weight("equal"),
                         select_fn=lambda s: select_sector_capped(
                             s, sectors, top_n=top_n, max_per_sector=max_per_sector),
                         lookback=lookback, rebalance=rebalance, adv=adv,
                         capital=capital)
    return {"broad equal-weight": _summ(bench, target_vol, train_end),
            "PEAD top-beats": _summ(pead, target_vol, train_end)}


def _print(res, title):
    print(f"\n{title}")
    print(f"{'book':20}{'FULL Sharpe':>13}{'FULL MaxDD':>12}"
          f"{'TEST Sharpe':>13}{'TEST MaxDD':>12}")
    for name, m in res.items():
        f, t = m["full"], m["test"]
        print(f"{name:20}{f['sharpe']:>13.3f}{f['max_dd']:>12.3f}"
              f"{t['sharpe']:>13.3f}{t['max_dd']:>12.3f}")


if __name__ == "__main__":
    import sys
    from earnings_data import fetch_surprises
    from equity_sentiment import names_ever_held

    names = names_ever_held()
    sp = fetch_surprises(names)

    # 1) apples-to-apples on the deployed book (mega-cap, quarterly)
    _print(run_compare_pit(sp, rebalance=63),
           "PIT liquid-30, QUARTERLY (deployed book):")
    # 2) fair shot: monthly rebalance catches the ~40-60d drift window
    _print(run_compare_pit(sp, rebalance=21),
           "PIT liquid-30, MONTHLY (catches drift window):")
