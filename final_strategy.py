"""
final_strategy.py — the LOCKED candidate strategy + its chart set.

LOCKED SPEC (result of the full research arc; honest = PIT/de-biased + realistic
costs; frozen 2026-08-02):
  universe   : point-in-time S&P 1500 membership (delisted names included).
  selection  : the 30 most-liquid members, capped at 5 per sector (no sector
               >~17-20%) — broad, liquid, diversified.
  weighting  : EQUAL WEIGHT (1/N) — beat every risk-based allocator out-of-sample.
  cadence    : quarterly (63 trading days) — low turnover (~4%).
  overlay    : 15% annualized VOLATILITY TARGET (de-risk into cash in turbulent
               periods; no leverage, no look-ahead). Lifts Sharpe, ~halves drawdown.

Performance (PIT, realistic costs): Sharpe ~0.92 (+vol-target), CAGR ~17%,
max drawdown ~-18%, held-out-test Sharpe strong. The sector cap removes the
tech-concentration tilt at no cost to risk-adjusted return, confirming the edge
comes from liquid + equal-weight + vol-target, not a sector bet.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import historical_membership as hm
from backtester import (walk_forward, benchmark_equal_weight, performance,
                        vol_target)
from sector_select import load_sectors, select_sector_capped, sector_breakdown
from costs import load_adv
from allocation_bakeoff import make_score_liquid, make_weight

# ---- LOCKED PARAMETERS ----
BASKET_N = 30            # most-liquid names held
MAX_PER_SECTOR = 5       # diversification cap (no sector > ~17-20%)
LOOKBACK = 252
REBALANCE = 63           # quarterly
CAPITAL = 5_000_000.0
TARGET_VOL = 0.15        # annualized vol-target overlay


def build(basket_n=BASKET_N, max_per_sector=MAX_PER_SECTOR,
          prices_path="prices_pit.pkl"):
    prices = pd.read_pickle(prices_path).dropna(how="all", axis=1).sort_index()
    membership = hm.load_membership()
    sectors = load_sectors("universe.csv")
    adv = load_adv("universe.csv")

    score_fn = hm.pit_score(make_score_liquid(adv), membership)
    select_fn = lambda s: select_sector_capped(
        s, sectors, top_n=basket_n, max_per_sector=max_per_sector)
    res = walk_forward(prices, score_fn, make_weight("equal"), select_fn=select_fn,
                       lookback=LOOKBACK, rebalance=REBALANCE,
                       adv=adv, capital=CAPITAL)

    raw = res["returns"]
    # vol-target scale (for exposure chart) + overlaid returns
    realized = raw.rolling(21).std() * np.sqrt(252)
    scale = (TARGET_VOL / realized).clip(upper=1.0).shift(1).fillna(1.0)
    overlaid = vol_target(raw, target=TARGET_VOL)

    _, bench = benchmark_equal_weight(prices, start_from=LOOKBACK)
    bench = bench.reindex(raw.index).fillna(0.0)
    return {"res": res, "raw": raw, "overlaid": overlaid, "scale": scale,
            "bench": bench, "sectors": sectors, "basket_n": basket_n}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def make_charts(basket_n=BASKET_N, max_per_sector=MAX_PER_SECTOR,
                outdir="figures_final"):
    import os
    d = build(basket_n, max_per_sector)
    os.makedirs(outdir, exist_ok=True)
    raw, ov, scale, bench = d["raw"], d["overlaid"], d["scale"], d["bench"]
    sectors = d["sectors"]
    tag = f"liquid-{basket_n} 1/N (cap {max_per_sector}/sector)"

    def eq(r):
        return (1 + r).cumprod()

    # 01 equity curves (log)
    fig, ax = plt.subplots(figsize=(10, 5))
    eq(raw).plot(ax=ax, label=f"{tag} raw", lw=1.3)
    eq(ov).plot(ax=ax, label=f"{tag} + 15% vol-target", lw=1.6)
    eq(bench).plot(ax=ax, label="equal-weight universe (bench)", lw=1.0, alpha=0.7)
    ax.set_yscale("log"); ax.set_title(f"Equity curve — {tag} (PIT, realistic costs)")
    ax.legend(); ax.set_ylabel("growth of $1 (log)")
    _save(fig, f"{outdir}/01_equity.png")

    # 02 drawdown underwater
    fig, ax = plt.subplots(figsize=(10, 4))
    for r, lab in [(raw, "raw"), (ov, "+vol-target")]:
        e = eq(r); dd = e / e.cummax() - 1
        dd.plot(ax=ax, label=lab, lw=1.2)
    ax.set_title(f"Drawdown — {tag}"); ax.legend(); ax.set_ylabel("drawdown")
    ax.fill_between(eq(ov).index, (eq(ov)/eq(ov).cummax()-1), 0, alpha=0.15)
    _save(fig, f"{outdir}/02_drawdown.png")

    # 03 rolling 63d annualized vol vs target
    fig, ax = plt.subplots(figsize=(10, 4))
    (raw.rolling(63).std()*np.sqrt(252)).plot(ax=ax, label="raw vol", lw=1.0)
    (ov.rolling(63).std()*np.sqrt(252)).plot(ax=ax, label="overlaid vol", lw=1.3)
    ax.axhline(TARGET_VOL, color="k", ls="--", lw=0.8, label="15% target")
    ax.set_title(f"Rolling 63d volatility — {tag}"); ax.legend(); ax.set_ylabel("ann. vol")
    _save(fig, f"{outdir}/03_rolling_vol.png")

    # 04 rolling 126d Sharpe
    fig, ax = plt.subplots(figsize=(10, 4))
    for r, lab in [(raw, "raw"), (ov, "+vol-target"), (bench, "bench")]:
        rs = (r.rolling(126).mean()*252 - 0.04) / (r.rolling(126).std()*np.sqrt(252))
        rs.plot(ax=ax, label=lab, lw=1.1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"Rolling 126d Sharpe — {tag}"); ax.legend()
    _save(fig, f"{outdir}/04_rolling_sharpe.png")

    # 05 market exposure (vol-target scale)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    scale.plot(ax=ax, lw=1.0)
    ax.fill_between(scale.index, scale.values, 0, alpha=0.2)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Market exposure from 15% vol-target — {tag} (1.0=fully invested)")
    ax.set_ylabel("exposure")
    _save(fig, f"{outdir}/05_exposure.png")

    # 06 current book weights
    last_w = d["res"]["weights"].iloc[-1]
    held = last_w[last_w > 1e-6].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    held.plot.bar(ax=ax)
    ax.set_title(f"Current book — {tag} ({len(held)} names, equal-weight)")
    ax.set_ylabel("weight")
    _save(fig, f"{outdir}/06_weights.png")

    # 07 sector breakdown
    br = sector_breakdown(list(held.index), sectors)
    fig, ax = plt.subplots(figsize=(9, 4))
    pd.Series(br).sort_values().plot.barh(ax=ax)
    ax.set_title(f"Sector breakdown — {tag}"); ax.set_xlabel("# names")
    _save(fig, f"{outdir}/07_sectors.png")

    # 08 annual returns
    ann_s = (1+ov).groupby(ov.index.year).prod() - 1
    ann_b = (1+bench).groupby(bench.index.year).prod() - 1
    fig, ax = plt.subplots(figsize=(10, 4))
    adf = pd.DataFrame({"strategy(+volTgt)": ann_s, "bench": ann_b})
    adf.plot.bar(ax=ax)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"Annual returns — {tag}"); ax.set_ylabel("return")
    _save(fig, f"{outdir}/08_annual_returns.png")

    m_raw, m_ov = performance(raw), performance(ov)
    print(f"[{tag}] wrote 8 charts to {outdir}/  | "
          f"raw Sharpe {m_raw['sharpe']:.2f} DD {m_raw['max_dd']*100:.0f}%  ->  "
          f"+volTgt Sharpe {m_ov['sharpe']:.2f} DD {m_ov['max_dd']*100:.0f}%")
    return d


if __name__ == "__main__":
    make_charts()
