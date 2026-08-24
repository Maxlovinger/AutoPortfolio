"""
fx_execution.py — turn the FX carry sleeve's signed USD notionals into concrete,
correctly-oriented IBKR spot-FX orders, and (opt-in) transmit them to TWS.

THE QUOTE-CONVENTION TRAP THIS MODULE EXISTS TO SOLVE
-----------------------------------------------------
Our signal speaks ONE language: a target is a signed USD notional where
  positive = LONG the foreign currency  (profit if it strengthens vs USD)
  negative = SHORT the foreign currency (profit if it weakens vs USD)

IBKR speaks a DIFFERENT language for every pair. On IDEALPRO each pair has a
fixed "base.quote" orientation and the order QUANTITY is ALWAYS in the BASE
currency:
  * EUR / GBP / AUD / NZD trade as  FOREIGN.USD  (base = foreign)
        -> quantity is in FOREIGN units;  LONG foreign  = BUY  the pair.
  * everything else (JPY, CHF, CAD, SEK, NOK, and every EM ccy) trades as
        USD.FOREIGN  (base = USD)
        -> quantity is in USD units;       LONG foreign  = SELL the pair.

Get this backwards and you silently double down the WRONG way. So we derive the
pair, side, and quantity from fx.data's `invert` flag — the SAME flag that
already encodes each pair's Yahoo orientation — giving one source of truth.

Two real-world frictions this module surfaces rather than hides:
  * IDEALPRO shows its tight institutional spread only at/above ~USD 25k per
    order. Smaller orders still fill but as wider "odd lots". We FLAG (not block)
    legs below that; a small book's per-leg FX notionals will trip it.
  * A couple of EM currencies (KRW, CLP) are non-deliverable (NDF) and are NOT
    spot-tradeable on IDEALPRO. We flag and skip them instead of sending a doomed
    order.

SAFETY: like the rest of the live stack, routing defaults to dry-run. Nothing is
transmitted unless the caller passes dry_run=False (and the FX opt-in in
portfolio_live).
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from fx.data import WIDE

# IDEALPRO shows its tight spread at/above ~USD 25k/order; below is an odd lot.
IDEALPRO_TIGHT_MIN = 25_000.0
# Non-deliverable EM currencies — no IDEALPRO spot market (NDF only). Skip them.
NON_TRADEABLE_SPOT = {"KRW", "CLP"}


def ib_pair(ccy: str, universe=WIDE) -> tuple[str, str]:
    """
    (base, quote) for the IDEALPRO contract of `ccy` vs USD, from the `invert`
    flag. invert=False -> FOREIGN.USD (base=foreign); invert=True -> USD.FOREIGN
    (base=USD). Raises for USD itself or unknown currencies.
    """
    if ccy == "USD":
        raise ValueError("USD is the funding leg, not a tradeable foreign ccy")
    cfg = universe.get(ccy)
    if cfg is None:
        raise KeyError(f"{ccy} not in FX universe")
    return ("USD", ccy) if cfg["invert"] else (ccy, "USD")


def plan_leg(ccy: str, usd_notional: float, spot_px: float,
             universe=WIDE, min_notional=IDEALPRO_TIGHT_MIN) -> dict | None:
    """
    Translate ONE signed-USD-notional target into a concrete IB FX order leg.

    usd_notional : signed; + = long the foreign ccy, - = short it.
    spot_px      : USD value of 1 unit of the foreign ccy (fx.data orientation).

    Returns a dict describing the order, or None if there is nothing to do
    (zero/near-zero target, bad price). The dict's `qty` is in the pair's BASE
    currency (what IB wants), and side/action already account for the pair
    orientation so callers never re-reason about convention.
    """
    if not np.isfinite(usd_notional) or abs(usd_notional) < 1.0:
        return None
    if not np.isfinite(spot_px) or spot_px <= 0:
        return None

    base, quote = ib_pair(ccy, universe)
    long_foreign = usd_notional > 0
    side = "LONG" if long_foreign else "SHORT"

    if base == ccy:
        # FOREIGN.USD : quantity is in foreign units; long foreign = BUY the pair.
        qty = abs(usd_notional) / spot_px
        action = "BUY" if long_foreign else "SELL"
    else:
        # USD.FOREIGN : quantity is in USD; long foreign = SELL the pair.
        qty = abs(usd_notional)
        action = "SELL" if long_foreign else "BUY"

    return {
        "ccy": ccy,
        "pair": f"{base}.{quote}",
        "base": base,
        "quote": quote,
        "side": side,
        "action": action,
        "qty": int(round(qty)),           # IB FX quantity = whole base-ccy units
        "usd_notional": round(float(usd_notional), 2),
        "spot": round(float(spot_px), 6),
        "below_min": abs(usd_notional) < min_notional,
        "tradeable": ccy not in NON_TRADEABLE_SPOT,
    }


def fx_order_plan(fx_targets: dict, spot_row: pd.Series, universe=WIDE,
                  min_notional=IDEALPRO_TIGHT_MIN) -> list[dict]:
    """
    Full order plan for the FX sleeve. `fx_targets` = {ccy: signed USD notional}
    (from portfolio_live.fx_sleeve_targets); `spot_row` = latest USD-per-foreign
    spot (fx.data d['spot'].iloc[-1]). Legs with no usable price or that are
    non-tradeable (NDF) are dropped from execution but reported via `below_min` /
    `tradeable` on each returned leg. Sorted long-first then by size.
    """
    legs = []
    for ccy, notional in fx_targets.items():
        px = float(spot_row.get(ccy, np.nan)) if ccy in spot_row.index else np.nan
        leg = plan_leg(ccy, notional, px, universe, min_notional)
        if leg is not None:
            legs.append(leg)
    legs.sort(key=lambda d: (-d["usd_notional"]))
    return legs


def summarize_plan(plan: list[dict]) -> dict:
    """Net/gross USD, and counts of the two friction flags — for previews/logs."""
    net = sum(l["usd_notional"] for l in plan)
    gross = sum(abs(l["usd_notional"]) for l in plan)
    return {
        "net_usd": round(net, 2),
        "gross_usd": round(gross, 2),
        "n_legs": len(plan),
        "n_below_min": sum(1 for l in plan if l["below_min"]),
        "n_non_tradeable": sum(1 for l in plan if not l["tradeable"]),
    }


def print_plan(plan: list[dict], min_notional=IDEALPRO_TIGHT_MIN):
    """Human-readable FX order plan with the convention made explicit."""
    s = summarize_plan(plan)
    print("\n  FX carry sleeve — ORDER PLAN (IDEALPRO spot):")
    for l in plan:
        flags = []
        if not l["tradeable"]:
            flags.append("NDF-SKIP")
        if l["below_min"]:
            flags.append("odd-lot")
        tag = ("  [" + ",".join(flags) + "]") if flags else ""
        print(f"    {l['side']:<5} {l['ccy']:<4}  {l['action']:<4} "
              f"{l['qty']:>12,} {l['pair']:<8} "
              f"(${abs(l['usd_notional']):>9,.0f}){tag}")
    print(f"    net ${s['net_usd']:,.0f} | gross ${s['gross_usd']:,.0f} | "
          f"{s['n_legs']} legs"
          + (f" | {s['n_below_min']} below IDEALPRO ${min_notional/1e3:.0f}k "
             f"(odd-lot spread)" if s['n_below_min'] else "")
          + (f" | {s['n_non_tradeable']} NDF skipped" if s['n_non_tradeable'] else ""))


# --- IBKR routing (opt-in transmit) ----------------------------------------
def route_fx(app, plan: list[dict], ibkr, dry_run=True, wait=25) -> list[dict]:
    """
    Send the FX order plan to TWS via ibapi (mirrors auto_rebalance's transmit +
    await-ack discipline). NDF (non-tradeable) legs are never sent. Returns the
    plan annotated with order_id/status. dry_run=True (default) sends nothing.
    """
    sendable = [l for l in plan if l["tradeable"] and l["qty"] > 0]
    if dry_run:
        for l in plan:
            l["status"] = "DRY-RUN"
        return plan

    oids = []
    for l in sendable:
        oid = app.next_order_id()
        contract = ibkr.forex(l["base"], l["quote"])
        # GTC: FX legs are odd-lots (< IDEALPRO $25k min) that often don't fill the
        # same session; DAY would expire them overnight (the book vanished this way).
        # GTC keeps an unfilled leg working until it fills, so the book converges.
        order = ibkr.market_order(l["action"], l["qty"], tif="GTC")
        app.placeOrder(oid, contract, order)
        l["order_id"] = oid
        oids.append(oid)

    # wait for TWS to acknowledge before the socket closes (else orders can drop)
    deadline = time.time() + wait
    while time.time() < deadline and not all(o in app.order_status for o in oids):
        time.sleep(0.5)
    for l in sendable:
        stt = app.order_status.get(l.get("order_id"))
        l["status"] = stt["status"] if stt else "UNCONFIRMED"
    for l in plan:
        if not l["tradeable"]:
            l["status"] = "SKIPPED-NDF"
    time.sleep(1.0)                        # final flush before caller disconnects
    return plan
