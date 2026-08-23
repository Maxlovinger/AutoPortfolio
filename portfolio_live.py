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


# --- whole-share rounding-drag top-up (pure, unit-tested) ------------------
# plan_orders FLOORs each name (nav*w // px), so on a small account the equity
# sleeve is left materially UNDER its target: the fractional remainders sit idle
# in cash AND any name priced above its per-name budget rounds to ZERO shares and
# is dropped entirely. At $50k this left equity ~13pp below target (43.6% vs the
# ~57% design). Fractional shares fix it exactly but need account permission; when
# they're unavailable this greedy top-up deploys the drag with WHOLE shares,
# staying as close to the equal-weight target as integer lots allow.
def greedy_share_topup(targets: dict, current_shares: dict, prices,
                       budget: float | None = None,
                       per_name_cap: float = 1.5) -> dict:
    """
    Largest-remainder whole-share fill toward per-name DOLLAR targets. Returns
    {ticker: extra_whole_shares} (BUYs only). Each step buys one share of the name
    that is currently MOST underweight in dollars, so tracking error to the target
    stays minimal. Rules:
      * never sells (only adds shares),
      * a name is capped at `per_name_cap` * its target so cheap names can't run
        away — EXCEPT a name currently at 0 shares may take its FIRST share even
        if that overshoots the cap, so names that plan_orders dropped still enter,
      * `budget` bounds total spend (default = the full remaining $ gap to target),
      * names with a non-finite / non-positive price are skipped.
    Pure: no I/O.
    """
    px = {t: float(prices.get(t, np.nan)) for t in targets}
    ok = {t: np.isfinite(px[t]) and px[t] > 0 for t in targets}
    add = {t: 0 for t in targets}
    val = {t: (float(current_shares.get(t, 0)) * px[t]) if ok[t] else np.nan
           for t in targets}

    if budget is None:
        # default budget = the total $ gap to target, but for a DROPPED name (0
        # shares held) reserve at least its first-share price so a name that costs
        # more than its own gap can still enter the book (the LLY/GS/GEV case).
        budget = 0.0
        for t in targets:
            if not ok[t] or val[t] >= targets[t]:
                continue
            need = targets[t] - val[t]
            if float(current_shares.get(t, 0)) <= 0:
                need = max(need, px[t])
            budget += need
    if not np.isfinite(budget) or budget <= 0:
        return {}

    spent = 0.0
    while True:
        cands = []
        for t in targets:
            if not ok[t] or val[t] >= targets[t]:
                continue
            if spent + px[t] > budget + 1e-9:          # doesn't fit remaining budget
                continue
            held0 = (float(current_shares.get(t, 0)) + add[t]) <= 0
            if val[t] + px[t] <= targets[t] * per_name_cap or held0:
                cands.append(t)
        if not cands:
            break
        t = max(cands, key=lambda t: targets[t] - val[t])   # most underweight ($)
        add[t] += 1
        val[t] += px[t]
        spent += px[t]
    return {t: n for t, n in add.items() if n > 0}


def merge_share_orders(orders: list, extra_shares: dict, prices) -> list:
    """Fold extra whole-share BUYs into an existing plan_orders list, summing
    share counts per ticker and dropping anything that nets to zero. Pure."""
    by_t = {o["ticker"]: dict(o) for o in orders}
    for t, n in extra_shares.items():
        if n == 0:
            continue
        if t in by_t:
            by_t[t]["shares"] += n
        else:
            by_t[t] = {"ticker": t, "shares": n, "action": "BUY",
                       "price": round(float(prices.get(t, np.nan)), 2)}
    out = []
    for o in by_t.values():
        if o["shares"] == 0:
            continue
        o["action"] = "BUY" if o["shares"] > 0 else "SELL"
        out.append(o)
    return out


def sleeve_breakdown(nav: float, equity_val: float, bond_val: float,
                     fx_gross: float) -> dict:
    """
    Split a live book into sleeve market values + % of NAV. The FX carry sleeve is
    dollar-neutral (it uses margin, not net cash), so its GROSS exposure is
    reported and cash is the remainder (NAV - equity - bonds). `equity_is_largest`
    flags whether equity is the biggest single sleeve — the "majority in equities"
    check. (In a heavily de-risked vol-target regime cash can legitimately exceed
    equity; this reports the state, it is not a hard invariant.)
    """
    cash = nav - equity_val - bond_val
    pct = lambda x: (x / nav) if nav else 0.0
    return {
        "nav": nav,
        "equity": {"value": equity_val, "pct": pct(equity_val)},
        "bonds": {"value": bond_val, "pct": pct(bond_val)},
        "fx_gross": {"value": fx_gross, "pct": pct(fx_gross)},
        "cash": {"value": cash, "pct": pct(cash)},
        "equity_is_largest": equity_val >= max(bond_val, fx_gross, cash),
        "equity_over_half": equity_val > 0.5 * nav,
    }


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


# --- live data pull + routing (all three sleeves) --------------------------
def live(kind="sim", dry_run=True, port=7497, nav_cap=None, fx_live=False,
         topup=True, topup_cap=1.5, verbose=True):
    """
    Pull live data, build the 3-sleeve target book, preview it, and route it.

    Sleeves & their safety gates:
      * STOCK sleeve (equity + IEF bonds) — routed through the SAME plan_orders +
        await-ack path the equity book already uses. Transmits when dry_run=False.
      * FX carry sleeve — long/short spot FX incl. EM. Built into a concrete,
        convention-correct IDEALPRO order plan and PREVIEWED always, but only
        transmitted when BOTH dry_run=False AND fx_live=True (a separate opt-in,
        because shorting EM spot has leverage/borrow/gap risk).

    So `dry_run=False` alone brings equity+bonds live; add `fx_live=True` to also
    send the currency book.
    """
    from data import download_prices
    from sector_select import load_sectors, load_names
    from costs import load_adv
    from paper_trader import plan_orders, make_broker
    import fx_execution as fxe
    import ibkr as ibkr_mod
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

    # carry sleeve signal + spot (latest); spot is needed to size FX legs.
    # LIVE carry uses FRESH BIS policy rates (the OECD 3M carry in load_all lags
    # 1-7 months — fine for backtests, too stale to trade). If the fresh pull
    # fails we still PREVIEW on the stale carry but BLOCK FX transmission.
    from fx.data import load_all, WIDE, fetch_policy_rates, policy_carry
    d = load_all(start="2015-01-01", universe=WIDE)
    spot_row = d["spot"].iloc[-1]
    carry_fresh = True
    try:
        pol = fetch_policy_rates(WIDE)
        carry_row = policy_carry(pol.iloc[-1])
        asof = pol.index[-1].date()
        print(f"\n  FX carry signal: FRESH BIS policy rates (as-of {asof}).")
    except Exception as e:
        carry_fresh = False
        carry_row = d["carry"].iloc[-1]
        print(f"\n  [WARN] fresh BIS policy rates unavailable ({e}); using STALE "
              f"OECD 3M carry — FX PREVIEW ONLY, transmission blocked.")

    broker = make_broker(kind, dry_run=dry_run, port=port)
    try:
        nav = broker.nav(latest) if kind != "ibkr" else broker.nav()
        budget = min(nav_cap, nav) if nav_cap else nav

        out = preview(budget, picks, exposure, carry_row, prices=latest)

        # ---- STOCK sleeve (equity + IEF) ----
        sw = out["stock_weights"]
        current = (ar.normalize_positions(broker._ibkr.positions(broker.app))
                   if kind == "ibkr" else {})
        stock_orders = plan_orders(sw, latest, budget, current)
        # whole-share plan_orders FLOORs, leaving the equity sleeve under target;
        # deploy that rounding drag with a largest-remainder whole-share top-up so
        # the book reaches its designed equity weight (skip if fractional shares
        # are ever used — those already hit the target exactly).
        if topup:
            projected = dict(current)
            for o in stock_orders:
                projected[o["ticker"]] = projected.get(o["ticker"], 0) + o["shares"]
            tgt = {t: float(sw[t]) * budget for t in sw.index}
            extra = greedy_share_topup(tgt, projected, latest, per_name_cap=topup_cap)
            if extra:
                stock_orders = merge_share_orders(stock_orders, extra, latest)
                print(f"  [top-up] deploying whole-share rounding drag into "
                      f"{len(extra)} name(s): {extra}")
        stock_live = (not dry_run) and kind == "ibkr"
        print(f"\n  STOCK-sleeve orders ({'LIVE' if stock_live else 'DRY-RUN'}): "
              f"{len(stock_orders)}")
        for o in stock_orders[:40]:
            print(f"    {o['action']:<4} {o['shares']:>6} {o['ticker']:<7} @ ${o['price']}")
        if stock_live:
            _transmit_stock(broker, ibkr_mod, stock_orders)

        # ---- FX carry sleeve ----
        plan = fxe.fx_order_plan(out["fx_targets"], spot_row)
        fxe.print_plan(plan)
        fx_transmit = (not dry_run) and kind == "ibkr" and fx_live and carry_fresh
        if fx_transmit:
            fxe.route_fx(broker.app, plan, ibkr_mod, dry_run=False)
            acks = sum(1 for l in plan if l.get("status") not in
                       (None, "UNCONFIRMED", "DRY-RUN", "SKIPPED-NDF"))
            print(f"  FX sleeve TRANSMITTED: {acks} legs acknowledged by TWS.")
        else:
            gate = ("stale carry (fresh rates unavailable)" if not carry_fresh
                    else "fx_live opt-in not set" if not fx_live else "dry-run")
            print(f"  FX sleeve NOT transmitted ({gate}).")
        out["stock_orders"], out["fx_plan"] = stock_orders, plan
    finally:
        if hasattr(broker, "disconnect"):
            broker.disconnect()
    return out


def _transmit_stock(broker, ibkr, orders, wait=25):
    """Send stock/ETF sleeve MKT orders and await TWS ack (as auto_rebalance does)."""
    import time
    oids = []
    for o in orders:
        oid = broker.app.next_order_id()
        broker.app.placeOrder(oid, ibkr.stock(o["ticker"]),
                              ibkr.market_order(o["action"], abs(o["shares"])))
        o["order_id"] = oid
        oids.append(oid)
    deadline = time.time() + wait
    while time.time() < deadline and not all(x in broker.app.order_status for x in oids):
        time.sleep(0.5)
    acks = sum(1 for x in oids if x in broker.app.order_status)
    time.sleep(1.0)
    print(f"  stock-sleeve orders transmitted: {acks}/{len(orders)} acknowledged.")


def main():
    import sys
    kind = "ibkr" if "--ibkr" in sys.argv else "sim"
    dry_run = "--live" not in sys.argv          # transmit only with explicit --live
    fx_live = "--fx-live" in sys.argv           # separate opt-in for the FX sleeve
    topup = "--no-topup" not in sys.argv        # deploy whole-share rounding drag
    port = 4002 if "--gateway" in sys.argv else 7497
    nav_cap = 40000 if kind == "sim" else None  # sim has no real account NAV
    if kind == "ibkr":
        eq = "LIVE" if not dry_run else "DRY-RUN"
        fx = "LIVE" if (not dry_run and fx_live) else "preview"
        print(f"3-sleeve live: stock(equity+IEF)={eq}  FX={fx}  port={port}")
    live(kind=kind, dry_run=dry_run, port=port, nav_cap=nav_cap, fx_live=fx_live,
         topup=topup)
    print("\nUsage: python3 portfolio_live.py [--ibkr] [--live] [--fx-live] "
          "[--gateway] [--no-topup]")
    print("  default = SimBroker dry-run. --ibkr routes to TWS; --live transmits the")
    print("  stock sleeve (equity+IEF); add --fx-live to ALSO transmit the FX book.")
    print("  --no-topup disables the whole-share rounding-drag redeployment.")


if __name__ == "__main__":
    main()
