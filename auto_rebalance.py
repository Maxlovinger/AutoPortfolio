"""
auto_rebalance.py — the scheduled, automated live job for the LOCKED strategy.

WHAT IT RUNS (see final_strategy.py / report.md for the full research trail):
  * Holdings: the 30 most-liquid eligible names, <=5 per sector, EQUAL WEIGHT.
  * Holdings rebalance cadence: QUARTERLY (~63 trading days).
  * Exposure overlay: 15% annual VOLATILITY TARGET, managed WEEKLY with a +/-10%
    no-trade band (the cadence that matched daily-continuous at half the turnover;
    quarterly-only exposure was proven useless — it left drawdowns at -34%).

WHY IT LOGS RATIONALE
  Because it runs unattended, every decision is written to decision_log.md (human
  readable) and decision_log.jsonl (structured), so you can always see WHY it did
  what it did: why these names, what the volatility reading was, why exposure did
  or didn't change, and every order.

SAFETY
  Defaults to DRY-RUN (computes + logs orders, transmits nothing). Pass --live to
  actually route paper orders to IBKR TWS. Holds state in auto_state.json so the
  weekly band and quarterly schedule persist across scheduled runs.
"""
from __future__ import annotations
import json
import os
import re
import datetime as dt
import numpy as np
import pandas as pd

STATE_FILE = "auto_state.json"
LOG_MD = "decision_log.md"
LOG_JSONL = "decision_log.jsonl"

# ---- LOCKED PARAMETERS (mirror final_strategy.py) ----
TRADE_CAPITAL = None         # None = deploy the FULL account NAV (100% invested,
                             # no structurally idle cash). Set to a $ number to
                             # instead cap deployment and leave the rest in cash.
                             # We reset the paper account to a small balance in
                             # Client Portal, so full-NAV == the whole ~$40k book.
BASKET_N = 30
MAX_PER_SECTOR = 5
TARGET_VOL = 0.15
EXPOSURE_BAND = 0.10          # +/-10% no-trade band
VOL_LOOKBACK = 21            # trailing days for realized vol
REBALANCE_DAYS = 63          # quarterly holdings rebalance
N_CANDIDATES = 80            # most-liquid names to pull live prices for


# ----------------------------------------------------------------------
# Pure decision logic (fully unit-tested, no I/O)
# ----------------------------------------------------------------------
def is_rebalance_day(today, last_rebalance, freq_days=REBALANCE_DAYS) -> bool:
    """Quarterly holdings rebalance (or the very first run)."""
    if last_rebalance is None:
        return True
    return (pd.Timestamp(today) - pd.Timestamp(last_rebalance)).days >= freq_days - 3


def is_exposure_check_day(today) -> bool:
    """Weekly exposure review — run it on Mondays (or first run)."""
    return pd.Timestamp(today).weekday() == 0


def realized_vol(returns: pd.Series, lookback=VOL_LOOKBACK) -> float:
    """Trailing annualized volatility of the book (uses only past data)."""
    r = returns.dropna().tail(lookback)
    if len(r) < max(5, lookback // 2):
        return float("nan")
    return float(r.std(ddof=0) * np.sqrt(252))


def decide_exposure(vol, held, target=TARGET_VOL, band=EXPOSURE_BAND):
    """Return (new_exposure, changed, rationale). Weekly banded vol-target."""
    if not np.isfinite(vol) or vol <= 0:
        return held, False, "volatility unavailable; holding exposure unchanged"
    target_exp = min(target / vol, 1.0)
    move = abs(target_exp - held)
    if move > band:
        return (round(target_exp, 3), True,
                f"realized vol {vol*100:.1f}% -> target exposure "
                f"{target_exp*100:.0f}%; moved {move*100:.0f}pts (> {band*100:.0f}% "
                f"band) so RE-SIZED from {held*100:.0f}% to {target_exp*100:.0f}%")
    return (held, False,
            f"realized vol {vol*100:.1f}% -> target exposure {target_exp*100:.0f}%; "
            f"only {move*100:.0f}pts from held {held*100:.0f}% (<= {band*100:.0f}% "
            f"band) so NO CHANGE")


def _base_company(name: str) -> str:
    """Collapse a share-class label to its underlying company, so dual-class
    tickers (GOOG/GOOGL 'Alphabet Inc. (Class C/A)', FOX/FOXA, NWS/NWSA) map to
    one name and we don't hold the same company twice."""
    return re.sub(r"\s*\(Class [A-Z]\)\s*$", "", str(name)).strip()


def select_book(adv: pd.Series, sectors: dict, valid: set,
                n=BASKET_N, cap=MAX_PER_SECTOR, names: dict | None = None) -> list[str]:
    """The 30 most-liquid *currently-priced* names, <= cap per sector, with at
    most ONE share class per company (keeps the most-liquid class, since we scan
    in descending-ADV order)."""
    ranked = adv[adv.index.isin(valid)].sort_values(ascending=False)
    picks, counts, seen_co = [], {}, set()
    for t in ranked.index:
        sec = sectors.get(t, "Unknown")
        if counts.get(sec, 0) >= cap:
            continue
        if names is not None:
            co = _base_company(names.get(t, t))
            if co in seen_co:          # already hold this company's other class
                continue
            seen_co.add(co)
        picks.append(t)
        counts[sec] = counts.get(sec, 0) + 1
        if len(picks) >= n:
            break
    return picks


def target_weights(picks, exposure) -> pd.Series:
    """Equal weight across picks, scaled by exposure (rest is cash)."""
    if not picks:
        return pd.Series(dtype=float)
    return pd.Series(exposure / len(picks), index=picks)


# ----------------------------------------------------------------------
# State + rationale logging
# ----------------------------------------------------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"held_exposure": 1.0, "last_rebalance": None,
            "last_holdings": [], "inception": dt.date.today().isoformat()}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def log_decision(record: dict, rationale_lines: list[str]):
    """Append a structured row and a human-readable block."""
    with open(LOG_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")
    with open(LOG_MD, "a") as f:
        f.write(f"\n## {record['date']} — {record['run_type']}\n\n")
        for line in rationale_lines:
            f.write(f"- {line}\n")
        if record.get("orders"):
            f.write("\n**Orders:**\n\n")
            for o in record["orders"]:
                f.write(f"    {o['action']:4s} {abs(o['shares']):>5d} "
                        f"{o['ticker']:6s} @ ~${o['price']}\n")
        else:
            f.write("\n*(no orders)*\n")
        f.write(f"\nNAV ${record.get('nav', 0):,.0f} | "
                f"exposure {record.get('exposure', 0)*100:.0f}% | "
                f"{len(record.get('holdings', []))} holdings\n")


# ----------------------------------------------------------------------
# The scheduled run
# ----------------------------------------------------------------------
def run(kind="sim", dry_run=True, port=7497, today=None, verbose=True, moo=False,
        force_exposure=None):
    from data import download_prices
    from sector_select import load_sectors, load_names
    from costs import load_adv
    from paper_trader import plan_orders, make_broker

    today = pd.Timestamp(today or dt.date.today())
    state = load_state()
    sectors = load_sectors("universe.csv")
    names = load_names("universe.csv")
    adv = load_adv("universe.csv")

    do_reb = is_rebalance_day(today, state["last_rebalance"])
    do_exp = is_exposure_check_day(today) or do_reb
    if not (do_reb or do_exp):
        if verbose:
            print(f"[{today.date()}] nothing scheduled (not a Monday or quarter-end).")
        return None

    # ---- live prices for the most-liquid candidates ----
    candidates = adv.sort_values(ascending=False).head(N_CANDIDATES).index.tolist()
    prices = download_prices(candidates, start=str((today - pd.Timedelta(days=400)).date()))
    if prices is None or prices.shape[1] == 0:
        raise RuntimeError("No live prices (yfinance issue). Retry shortly.")
    latest = prices.iloc[-1]
    valid = {t for t in prices.columns if np.isfinite(latest.get(t, np.nan))}

    # ---- holdings: keep last quarter's unless it's a rebalance day ----
    rationale = [f"Run type: {'QUARTERLY REBALANCE + ' if do_reb else ''}"
                 f"{'weekly exposure check' if do_exp else ''}."]
    if do_reb:
        picks = select_book(adv, sectors, valid, names=names)
        prev = set(state.get("last_holdings", []))
        added = [t for t in picks if t not in prev]
        dropped = [t for t in prev if t not in picks]
        rationale.append(f"Selected the {len(picks)} most-liquid eligible names, "
                         f"capped {MAX_PER_SECTOR}/sector, equal-weight.")
        if prev:
            rationale.append(f"Changes vs last quarter — added {added or 'none'}; "
                             f"dropped {dropped or 'none'} (liquidity re-ranking).")
    else:
        picks = [t for t in state.get("last_holdings", []) if t in valid]
        rationale.append("Not a rebalance day — holdings unchanged; only exposure "
                         "reviewed.")

    # ---- exposure: weekly banded vol-target ----
    book_prices = prices[[t for t in picks if t in prices.columns]]
    book_ret = book_prices.pct_change(fill_method=None).mean(axis=1)  # equal-weight
    vol = realized_vol(book_ret)
    if force_exposure is not None:
        # Manual override: pin exposure (e.g. an off-cadence holdings fix where we
        # deliberately DON'T re-run the weekly vol-target review).
        held = float(state["held_exposure"])
        new_exp, changed = float(force_exposure), float(force_exposure) != held
        exp_reason = (f"PINNED to {new_exp*100:.0f}% (manual override; weekly "
                      f"vol-target review left for the scheduled Monday run).")
    else:
        new_exp, changed, exp_reason = decide_exposure(vol, float(state["held_exposure"]))
    rationale.append("Exposure: " + exp_reason)
    state["held_exposure"] = new_exp

    weights = target_weights(picks, new_exp)

    # ---- route orders ----
    broker = make_broker(kind, dry_run=dry_run, port=port)
    try:
        nav = broker.nav(latest) if kind != "ibkr" else broker.nav()
        # deploy only TRADE_CAPITAL (rest of the account stays in cash)
        if TRADE_CAPITAL and nav > 0:
            budget = min(TRADE_CAPITAL, nav)
            weights = weights * (budget / nav)
            rationale.append(f"Deploying ${budget:,.0f} of ${nav:,.0f} account "
                             f"(TRADE_CAPITAL cap); rest stays cash.")
        if kind == "ibkr":
            import time
            from paper_trader import plan_orders as _po
            # IBKR reports class shares with a SPACE ("BRK B"); our target names
            # use a DASH ("BRK-B"). Normalize so held shares reconcile against
            # targets — otherwise every rebalance re-buys them as if unheld.
            current = {k.replace(" ", "-"): int(v)
                       for k, v in broker._ibkr.positions(broker.app).items()}
            orders = _po(weights, latest, nav, current)
            if not dry_run:
                order_builder = (broker._ibkr.market_on_open_order if moo
                                 else broker._ibkr.market_order)
                for o in orders:
                    oid = broker.app.next_order_id()
                    broker.app.placeOrder(oid, broker._ibkr.stock(o["ticker"]),
                                          order_builder(o["action"],
                                                        abs(o["shares"])))
                    o["order_id"] = oid
                # WAIT for TWS to acknowledge before disconnecting (else the async
                # orders can be dropped when the socket closes). Poll orderStatus.
                oids = [o["order_id"] for o in orders]
                deadline = time.time() + 25
                while time.time() < deadline and not all(
                        oid in broker.app.order_status for oid in oids):
                    time.sleep(0.5)
                for o in orders:
                    stt = broker.app.order_status.get(o["order_id"])
                    o["status"] = stt["status"] if stt else "UNCONFIRMED"
                acks = sum(1 for o in orders if o.get("status") != "UNCONFIRMED")
                errs = [f"{c}: {t}" for lvl, c, t in broker.app.messages
                        if lvl == "error"]
                rationale.append(f"Transmitted {len(orders)} orders; {acks} "
                                 f"acknowledged by TWS.")
                if errs:
                    rationale.append("TWS errors: " + "; ".join(errs[:6]))
                time.sleep(1.0)          # final flush before disconnect
        else:
            trades = broker.rebalance(weights, latest)
            nav = broker.snapshot(latest)
            broker.save()
            orders = [{"ticker": t["ticker"], "shares": t["shares"],
                       "action": "BUY" if t["shares"] > 0 else "SELL",
                       "price": t["price"]} for t in trades]
    finally:
        if hasattr(broker, "disconnect"):
            broker.disconnect()

    if do_reb:
        state["last_rebalance"] = today.date().isoformat()
        state["last_holdings"] = picks
    save_state(state)

    record = {"date": today.date().isoformat(),
              "run_type": ("QUARTERLY REBALANCE" if do_reb else "weekly exposure check")
              + (" [DRY-RUN]" if dry_run else " [LIVE]"),
              "exposure": new_exp, "exposure_changed": changed,
              "realized_vol": None if not np.isfinite(vol) else round(vol, 4),
              "holdings": picks, "nav": round(float(nav), 2), "orders": orders}
    log_decision(record, rationale)

    if verbose:
        print(f"[{today.date()}] {record['run_type']}")
        for line in rationale:
            print("  - " + line)
        print(f"  NAV ${nav:,.0f} | exposure {new_exp*100:.0f}% | "
              f"{len(picks)} holdings | {len(orders)} orders")
        print(f"  Rationale appended to {LOG_MD}")
    return record


def main():
    import sys
    kind = "ibkr" if "--ibkr" in sys.argv else "sim"
    dry_run = "--live" not in sys.argv
    port = 4002 if "--gateway" in sys.argv else 7497
    moo = "--moo" in sys.argv          # Market-on-Open: stage orders for next open
    run(kind=kind, dry_run=dry_run, port=port, moo=moo)
    print("\nUsage: python3 auto_rebalance.py [--ibkr] [--live] [--gateway]")
    print("  default = SimBroker dry-run. --ibkr routes to TWS. --live transmits.")


if __name__ == "__main__":
    main()
