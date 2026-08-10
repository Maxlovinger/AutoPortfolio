"""
portfolio_live.py — the 3-SLEEVE live portfolio: equity + FX carry + IEF bonds.

Locked design allocation (capital weights):
    Equity   63.9%   (0.90 * 71%)  — 30 liquid US stocks, vol-targeted
    Currency 26.1%   (0.90 * 29%)  — long high-carry / short low-carry (G10+EM)
    Bonds    10.0%                 — IEF (7-10y US Treasuries)

Execution split by RISK:
  * STOCK sleeve (equity + IEF) — all LONG stocks/ETFs, routed through the SAME
    battle-tested plan_orders/broker path the equity book already uses. Live-ready.
  * FX sleeve — long/short spot FX incl. EM. Computed and PREVIEWED here, but NOT
    auto-transmitted: shorting EM currencies live has leverage/borrow/gap risk and
    quote-convention pitfalls that need an explicit, deliberate opt-in + review.
    (ibkr.forex() exists for when that path is built.)

Key sizing rule: the equity sleeve is scaled by BOTH its 63.9% weight AND the
vol-target exposure; the de-risked equity slice goes to CASH (we proved routing
it into bonds/commodities loses). Bonds stay a fixed 10% (their own sleeve).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# locked design (see memory / bond sweep). equity/currency keep 71/29 in the 90%.
WEIGHTS = {"equity": 0.90 * 0.71, "currency": 0.90 * 0.29, "bonds": 0.10}
BOND_ETF = "IEF"


# --- pure allocation logic (unit-tested) -----------------------------------
def allocation(nav: float, weights=WEIGHTS) -> dict:
    """Dollar capital per sleeve."""
    return {k: round(nav * v, 2) for k, v in weights.items()}


def stock_sleeve_weights(equity_picks, exposure: float, weights=WEIGHTS,
                         bond_etf=BOND_ETF) -> pd.Series:
    """
    Combined LONG weight vector for the stock/ETF sleeve (equity + IEF), ready for
    plan_orders. Equity names are equal-weight, scaled by the equity sleeve weight
    AND the vol-target exposure (de-risked slice -> cash). IEF is a fixed weight,
    NOT scaled by equity exposure (separate sleeve).
    """
    w = {}
    n = len(equity_picks)
    if n:
        per = weights["equity"] * exposure / n
        for t in equity_picks:
            w[t] = w.get(t, 0.0) + per
    w[bond_etf] = w.get(bond_etf, 0.0) + weights["bonds"]
    return pd.Series(w, dtype=float)


def fx_sleeve_targets(carry_row: pd.Series, nav: float, n_long=3, n_short=3,
                      weights=WEIGHTS) -> dict:
    """
    Dollar-neutral carry book at the FX sleeve's gross exposure. Returns
    {currency: signed USD notional} — positive = long, negative = short. Gross
    |notional| sums to weights['currency']*nav; net ~ 0.
    """
    from fx.backtest_carry import carry_weights
    w = carry_weights(carry_row, n_long, n_short, gross=weights["currency"])
    return {c: round(float(w[c]) * nav, 2)
            for c in w.index if abs(float(w[c])) > 1e-12}


# --- preview (pure; prints the full target book) ---------------------------
def preview(nav, equity_picks, exposure, carry_row, prices=None,
            n_long=3, n_short=3):
    alloc = allocation(nav)
    sw = stock_sleeve_weights(equity_picks, exposure)
    fx = fx_sleeve_targets(carry_row, nav, n_long, n_short)
    cash = nav * (1 - sw.sum() - WEIGHTS["currency"])   # de-risked equity + rounding

    print(f"\n=== 3-SLEEVE TARGET BOOK  (NAV ${nav:,.0f}) ===")
    print(f"  sleeve capital:  Equity ${alloc['equity']:,.0f} "
          f"({WEIGHTS['equity']*100:.1f}%) | Currency ${alloc['currency']:,.0f} "
          f"({WEIGHTS['currency']*100:.1f}%) | Bonds ${alloc['bonds']:,.0f} "
          f"({WEIGHTS['bonds']*100:.0f}%)")
    print(f"  equity vol-target exposure: {exposure*100:.0f}%  "
          f"(de-risked slice -> cash ${cash:,.0f})")

    print("\n  STOCK sleeve (equity + IEF) — LIVE-READY (long ETFs/stocks):")
    for t, w in sw.sort_values(ascending=False).items():
        line = f"    {t:<7} {w*100:>5.2f}%  ${nav*w:>10,.0f}"
        if prices is not None and t in prices.index and np.isfinite(prices[t]):
            line += f"   ~{int(nav*w // prices[t])} sh @ ${prices[t]:.2f}"
        print(line)

    print("\n  FX carry sleeve — PREVIEW ONLY (not transmitted; long/short + EM):")
    for c, amt in sorted(fx.items(), key=lambda kv: kv[1]):
        side = "LONG " if amt > 0 else "SHORT"
        print(f"    {side} {c:<4} ${abs(amt):>9,.0f}")
    net = sum(fx.values())
    print(f"    (dollar-neutral: net ${net:,.0f}, gross ${sum(abs(v) for v in fx.values()):,.0f})")
    return {"stock_weights": sw, "fx_targets": fx, "allocation": alloc, "cash": cash}


# --- live data pull + dry-run (equity+IEF executable; FX preview) ----------
def live(kind="sim", dry_run=True, port=7497, nav_cap=None, verbose=True):
    """
    Pull live data, build the 3-sleeve target book, preview it, and route ONLY the
    stock sleeve (equity+IEF) through the broker (dry-run by default). FX is
    previewed, never auto-sent.
    """
    from data import download_prices
    from sector_select import load_sectors, load_names
    from costs import load_adv
    from paper_trader import plan_orders, make_broker
    import auto_rebalance as ar

    sectors, names, adv = (load_sectors("universe.csv"), load_names("universe.csv"),
                           load_adv("universe.csv"))
    cands = adv.sort_values(ascending=False).head(ar.N_CANDIDATES).index.tolist()
    prices = download_prices(cands + [BOND_ETF],
                             start=str((pd.Timestamp.today() - pd.Timedelta(days=400)).date()))
    latest = prices.iloc[-1]
    valid = {t for t in prices.columns if np.isfinite(latest.get(t, np.nan))}
    picks = ar.select_book(adv, sectors, valid, names=names)

    # exposure from the current book's realized vol (same as auto_rebalance)
    book_ret = prices[[t for t in picks if t in prices.columns]].pct_change(
        fill_method=None).mean(axis=1)
    exposure, _, _ = ar.decide_exposure(ar.realized_vol(book_ret), 1.0)

    # carry sleeve signal (latest month)
    from fx.data import load_all, WIDE
    d = load_all(start="2010-01-01", universe=WIDE)
    carry_row = d["carry"].iloc[-1]

    broker = make_broker(kind, dry_run=dry_run, port=port)
    nav = broker.nav(latest) if kind != "ibkr" else broker.nav()
    budget = min(nav_cap, nav) if nav_cap else nav

    out = preview(budget, picks, exposure, carry_row, prices=latest)

    # route the STOCK sleeve only (equity + IEF) — live-ready path
    sw = out["stock_weights"]
    current = {}
    if kind == "ibkr":
        current = ar.normalize_positions(broker._ibkr.positions(broker.app))
    orders = plan_orders(sw, latest, budget, current)
    print(f"\n  STOCK-sleeve orders ({'DRY-RUN' if dry_run else 'LIVE'}): {len(orders)}")
    for o in orders[:40]:
        print(f"    {o['action']:<4} {o['shares']:>6} {o['ticker']:<7} @ ${o['price']}")
    if not dry_run and kind == "ibkr":
        for o in orders:
            broker.place(o)
        print("  stock-sleeve orders transmitted.")
    print("\n  FX sleeve NOT transmitted (preview only) — wire live FX deliberately.")
    return out


if __name__ == "__main__":
    import sys
    live(kind="sim", dry_run=True, nav_cap=40000)
