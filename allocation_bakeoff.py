"""
allocation_bakeoff.py — compare risk-based weighting schemes on the PIT universe.

Motivation: our signal-picking backtest lost to a plain equal-weight benchmark
once bias + costs were honest. So the real question is HOW to weight a diversified
basket. This bakes off the allocators in allocators.py head-to-head, all on the
same point-in-time universe, same realistic costs, same rebalance schedule.

SELECTION (held fixed so we isolate the WEIGHTING effect):
  the lowest-volatility N members at each rebalance (the low-vol anomaly — the
  most robust price-only equity tilt), sector-capped for spread. Quarterly
  rebalance keeps turnover — and therefore costs — low, which is where these
  risk-based schemes shine.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import historical_membership as hm
from backtester import walk_forward, benchmark_equal_weight, performance
from sector_select import load_sectors, select_sector_capped
from costs import load_adv
from allocators import allocate

LOOKBACK = 252
REBALANCE = 63           # quarterly
CAPITAL = 5_000_000.0
BASKET_N = 60
MAX_PER_SECTOR = 10
VOL_LB = 126


def score_lowvol(window):
    """Higher score = lower trailing volatility (the names we want to hold)."""
    rets = window.pct_change(fill_method=None).tail(VOL_LB)
    return (-rets.std()).dropna()


def make_score_liquid(adv):
    """Broad, heterogeneous basket: most-liquid names (any vol) — gives the
    risk-based allocators real dispersion to work with. Fair test of WEIGHTING."""
    def _s(window):
        names = [c for c in window.columns if pd.notna(window[c].iloc[-1])]
        return adv.reindex(names).dropna()
    return _s


def make_weight(method, cap=0.10, min_obs=0.6, shrink=False):
    def _w(window, picks):
        win = window[picks].tail(LOOKBACK)
        good = [c for c in picks
                if win[c].notna().mean() >= min_obs and pd.notna(window[c].iloc[-1])]
        if len(good) < 3:
            good = [c for c in picks if pd.notna(window[c].iloc[-1])] or list(picks)
            return pd.Series(1.0 / len(good), index=good)
        rets = win[good].pct_change(fill_method=None).iloc[1:]
        return allocate(rets, method, cap=cap, shrink=shrink)
    return _w


def run(prices_path="prices_pit.pkl", methods=("equal", "inverse_vol",
                                               "min_var", "erc", "max_div", "hrp"),
        selection="lowvol", shrink=False, basket_n=BASKET_N, verbose=True):
    prices = pd.read_pickle(prices_path).dropna(how="all", axis=1).sort_index()
    membership = hm.load_membership()
    sectors = load_sectors("universe.csv")
    adv = load_adv("universe.csv")

    base_score = score_lowvol if selection == "lowvol" else make_score_liquid(adv)
    score_fn = hm.pit_score(base_score, membership)
    select_fn = lambda s: select_sector_capped(s, sectors, top_n=basket_n,
                                                max_per_sector=max(basket_n // 6, 10))

    _, bench = benchmark_equal_weight(prices, start_from=LOOKBACK)

    rows = {}
    for m in methods:
        res = walk_forward(prices, score_fn, make_weight(m, shrink=shrink),
                           select_fn=select_fn, lookback=LOOKBACK,
                           rebalance=REBALANCE, adv=adv, capital=CAPITAL)
        b = bench.reindex(res["returns"].index).fillna(0.0)
        perf = performance(res["returns"])
        # split-sample stability: Sharpe on held-out test (2025→)
        test = performance(res["returns"].loc["2024-12-31":])
        rows[m] = {"CAGR": perf["cagr"], "Vol": perf["vol"],
                   "Sharpe": perf["sharpe"], "MaxDD": perf["max_dd"],
                   "Test Sharpe": test["sharpe"],
                   "Turnover": res["turnover"].mean()}
    # benchmark row
    bp = performance(bench.reindex(
        walk_forward(prices, score_fn, make_weight("equal"), select_fn=select_fn,
                     lookback=LOOKBACK, rebalance=REBALANCE)["returns"].index).fillna(0.0))
    rows["BENCH 1/N univ"] = {"CAGR": bp["cagr"], "Vol": bp["vol"],
                              "Sharpe": bp["sharpe"], "MaxDD": bp["max_dd"],
                              "Test Sharpe": np.nan, "Turnover": np.nan}
    table = pd.DataFrame(rows).T
    if verbose:
        disp = table.copy()
        for c in ("CAGR", "Vol", "MaxDD"):
            disp[c] = (disp[c] * 100).map(lambda x: f"{x:6.2f}%")
        for c in ("Sharpe", "Test Sharpe"):
            disp[c] = disp[c].map(lambda x: f"{x:5.2f}" if np.isfinite(x) else "   —")
        disp["Turnover"] = disp["Turnover"].map(
            lambda x: f"{x*100:4.0f}%" if np.isfinite(x) else "  —")
        print(f"ALLOCATION BAKE-OFF — {selection} {basket_n}, quarterly, PIT "
              f"universe, realistic costs, shrink={shrink}\n")
        print(disp.to_string())
    return table


if __name__ == "__main__":
    run()
