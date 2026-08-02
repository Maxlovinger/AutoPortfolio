"""
strategy.py — the LOCKED strategy spec + disciplined out-of-sample evaluation.

THE LOCKED SPEC (frozen; no per-run tuning):
  universe   : point-in-time S&P 1500 membership (historical_membership.py) — the
               backtest can only pick names that were ACTUALLY in the index then,
               delisted names included where yfinance still has prices.
  signal     : short-term REVERSAL, sector-neutralized (z-scored within GICS
               sector) — the strongest honest factor in our IC study (IC IR 0.43).
  selection  : top-N with a per-sector cap (spread across industries).
  allocation : capped MIN-VARIANCE (trustworthy left-tip of the frontier;
               max-Sharpe tangency overfits in-sample).
  cadence    : monthly (21 trading days).
  costs      : realistic per-name spread + square-root impact (costs.py), $5M book.

DISCIPLINE:
  Timeline split TRAIN / VALIDATION / held-out TEST. Parameters are fixed by
  design (reversal-led, sensible defaults) — NOT optimized against the test
  slice. We report all three slices to judge stability and look at TEST once.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import historical_membership as hm
from backtester import (walk_forward, weight_min_variance, score_reversal,
                        performance, benchmark_equal_weight)
from sector_select import (load_sectors, sector_neutralize,
                           select_sector_capped, sector_breakdown)
from costs import load_adv

# ---- LOCKED PARAMETERS ----
REV_LOOKBACK = 21        # reversal signal window (days)
TOP_N = 20               # names held
MAX_PER_SECTOR = 3       # diversification cap
NAME_CAP = 0.15          # per-name weight cap
COV_LOOKBACK = 252       # trailing window for covariance
REBALANCE = 21           # monthly
CAPITAL = 5_000_000.0    # assumed book size (sets impact scale)

# ---- Split points (train / validation / held-out test) ----
TRAIN_END = "2022-12-31"
VAL_END = "2024-12-31"


def build_strategy(sectors, membership):
    """Return (score_fn, select_fn, weight_fn) for the locked spec."""
    def raw_score(window):
        rev = score_reversal(window, lookback=REV_LOOKBACK)
        return sector_neutralize(rev, sectors)

    score_fn = hm.pit_score(raw_score, membership)
    select_fn = lambda s: select_sector_capped(s, sectors, top_n=TOP_N,
                                                max_per_sector=MAX_PER_SECTOR)
    weight_fn = lambda w, p: weight_min_variance(w, p, cap=NAME_CAP,
                                                 lookback=COV_LOOKBACK)
    return score_fn, select_fn, weight_fn


def slice_metrics(returns, bench):
    """Performance on train / validation / held-out test, plus full period."""
    r = returns
    splits = {
        "TRAIN   (→2022)": r.loc[:TRAIN_END],
        "VALID   (23-24)": r.loc[TRAIN_END:VAL_END],
        "TEST    (25→ )":  r.loc[VAL_END:],
        "FULL":            r,
    }
    rows = {}
    for name, seg in splits.items():
        m = performance(seg)
        b = performance(bench.reindex(seg.index).fillna(0.0))
        rows[name] = {"CAGR": m["cagr"], "Vol": m["vol"], "Sharpe": m["sharpe"],
                      "MaxDD": m["max_dd"], "Bench Sharpe": b["sharpe"]}
    return pd.DataFrame(rows).T


def run(prices_path="prices_pit.pkl", verbose=True):
    prices = pd.read_pickle(prices_path).dropna(how="all", axis=1).sort_index()
    membership = hm.load_membership()
    sectors = load_sectors("universe.csv")
    adv = load_adv("universe.csv")

    score_fn, select_fn, weight_fn = build_strategy(sectors, membership)
    res = walk_forward(prices, score_fn, weight_fn,
                       select_fn=select_fn, lookback=COV_LOOKBACK,
                       rebalance=REBALANCE, adv=adv, capital=CAPITAL)
    _, bench = benchmark_equal_weight(prices, start_from=COV_LOOKBACK)
    bench = bench.reindex(res["returns"].index).fillna(0.0)

    table = slice_metrics(res["returns"], bench)
    if verbose:
        pd.set_option("display.float_format", lambda x: f"{x:.3f}")
        print("LOCKED STRATEGY — reversal-led, sector-neutral, min-variance, "
              "monthly, PIT universe, realistic costs\n")
        disp = table.copy()
        for c in ("CAGR", "Vol", "MaxDD"):
            disp[c] = (disp[c] * 100).map(lambda x: f"{x:6.2f}%")
        disp["Sharpe"] = disp["Sharpe"].map(lambda x: f"{x:5.2f}")
        disp["Bench Sharpe"] = disp["Bench Sharpe"].map(lambda x: f"{x:5.2f}")
        print(disp.to_string())
        print(f"\nAvg turnover/rebalance: {res['turnover'].mean()*100:.1f}%  "
              f"({res['weights'].shape[0]} rebalances)")
        # current book
        last_w = res["weights"].iloc[-1]
        held = last_w[last_w > 0].sort_values(ascending=False)
        print(f"\nMost recent book ({len(held)} names):")
        for t, w in held.items():
            print(f"  {t:6s} {w*100:5.1f}%  [{sectors.get(t,'?')}]")
        print("sectors:", sector_breakdown(list(held.index), sectors))
    return res, table


if __name__ == "__main__":
    run()
