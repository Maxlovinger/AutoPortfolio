"""
run_fx.py — FX carry sleeve as a standalone, cron-safe job.

Sizes the dollar-neutral carry book to the currency sleeve weight (26.1% of live NAV)
using FRESH BIS policy-rate carry, then RECONCILES against currently-held spot-FX:
each currency's current signed USD exposure is read from the portfolio (marketValue,
already USD), subtracted from target, and only the DELTA is traded. Idempotent — a
re-run already on target sends ~nothing (safe to cron monthly; won't stack positions).

Self-contained ibapi app (the shared IBKRClient stores positions as {symbol:qty},
which loses FX currency/secType — inadequate for netting USD.X pairs). Reuses the
proven fx_execution planner + ibkr contract helpers.

Cadence: monthly. Default DRY-RUN; --live transmits. Usage: python3 run_fx.py [--live] [--gateway]
"""
from __future__ import annotations
import sys, time, threading
from pathlib import Path
import numpy as np
import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ibkr as ibkr_mod
import fx_execution as fxe
import portfolio_live as pl
from fx.data import WIDE, load_all, fetch_policy_rates, policy_carry

LIVE = "--live" in sys.argv
PORT = 4002 if "--gateway" in sys.argv else 7497
LOG = Path(__file__).resolve().parent / "fx_run.log"


def _argval(flag, default):
    """Read an int flag like --nlong 2 from argv (fallback ladder for leg count/size)."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                pass
    return default


N_LONG = _argval("--nlong", 3)     # 3+3 default; drop to 2+2 or 1+1 for bigger legs
N_SHORT = _argval("--nshort", 3)   # (each leg must clear IDEALPRO's ~$25k min to persist)
INFO = {2104, 2106, 2107, 2108, 2119, 2158, 2100, 2150, 2119}


def log(msg):
    line = f"{pd.Timestamp.now(tz='UTC').isoformat(timespec='seconds')}  {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line)


def held_to_signed(raw_fx, spot_row) -> dict:
    """
    Convert held spot-FX positions to {ccy: signed USD notional, + = long foreign}.
    Pure (unit-tested) — this is the reconciliation core that must invert IB's
    per-pair quote convention correctly, or a monthly re-run trades the wrong way.
      * USD.FOREIGN pair (symbol == 'USD'): position is in USD; long USD = SHORT
        the foreign ccy -> signed notional = -position.
      * FOREIGN.USD pair (symbol == foreign): position is in foreign units ->
        signed USD notional = position * spot (USD per foreign unit).
    """
    current = {}
    for sym, cur, pos in raw_fx:
        if sym == "USD":
            current[cur] = current.get(cur, 0.0) - float(pos)
        else:
            px = float(spot_row.get(sym, np.nan))
            if np.isfinite(px):
                current[sym] = current.get(sym, 0.0) + float(pos) * px
    return current


def reconcile_delta(target: dict, current: dict, min_usd=50.0) -> dict:
    """Per-currency delta = target - current, dropping legs below min_usd (no-churn band)."""
    ccys = set(target) | set(current)
    delta = {c: round(target.get(c, 0.0) - current.get(c, 0.0), 2) for c in ccys}
    return {c: v for c, v in delta.items() if abs(v) >= min_usd}


class App(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self._next = None; self._idev = threading.Event()
        self.nlv = None; self.raw_fx = []      # (symbol, currency, position) for held CASH
        self.order_status = {}; self.acctdone = False; self.posdone = False
    def nextValidId(self, oid): self._next = oid; self._idev.set()
    def next_order_id(self):
        o = self._next; self._next += 1; return o
    def error(self, reqId, code="", msg="", *a):
        if code not in INFO: log(f"IB msg {code}: {msg}")
    def updateAccountValue(self, k, v, cur, acct):
        if k == "NetLiquidation" and cur in ("USD", "BASE"):
            self.nlv = float(v)
    def accountDownloadEnd(self, a): self.acctdone = True
    # held positions via reqPositions (reliable for FX; portfolio/marketValue is not
    # populated for FX on a closed-market weekend, which silently zeroed reconciliation)
    def position(self, acct, c, position, avgCost):
        if c.secType == "CASH" and position != 0:
            self.raw_fx.append((c.symbol, c.currency, float(position)))
    def positionEnd(self): self.posdone = True
    def orderStatus(self, oid, status, *a):
        self.order_status[oid] = {"status": status}


def main():
    app = App(); app.connect("127.0.0.1", PORT, clientId=17)
    threading.Thread(target=app.run, daemon=True).start()
    if not app._idev.wait(10):
        log("ABORT: no connection to Gateway."); return
    try:
        app.reqAccountUpdates(True, "")
        app.reqPositions()
        t = time.time()
        while (not app.acctdone or not app.posdone) and time.time() - t < 20:
            time.sleep(0.3)
        time.sleep(1)
        nav = app.nlv or 0.0
        if nav <= 0:
            log("ABORT: could not read NAV."); return

        d = load_all(start="2015-01-01", universe=WIDE)
        spot_row = d["spot"].iloc[-1]
        try:
            pol = fetch_policy_rates(WIDE)
            carry_row = policy_carry(pol.iloc[-1])
            log(f"FX carry signal: FRESH BIS policy rates as-of {pol.index[-1].date()}")
        except Exception as e:
            log(f"ABORT: fresh BIS policy rates unavailable ({e}); refusing to trade on stale carry.")
            return

        target = pl.fx_sleeve_targets(carry_row, nav, n_long=N_LONG, n_short=N_SHORT)
        log(f"FX book: {N_LONG} long / {N_SHORT} short legs")
        current = held_to_signed(app.raw_fx, spot_row)
        delta = reconcile_delta(target, current, min_usd=50.0)   # no-churn band

        gross = sum(abs(v) for v in target.values())
        log(f"NAV ${nav:,.0f} | currency sleeve gross target ${gross:,.0f}")
        log(f"target : { {c: round(v) for c, v in sorted(target.items())} }")
        log(f"current: { {c: round(v) for c, v in sorted(current.items())} }")
        log(f"delta  : { {c: round(v) for c, v in sorted(delta.items())} }")

        plan = fxe.fx_order_plan(delta, spot_row)
        fxe.print_plan(plan)

        if not LIVE:
            log("DRY-RUN — nothing transmitted."); return
        if not plan:
            log("Book already on target — no FX orders needed."); return
        fxe.route_fx(app, plan, ibkr_mod, dry_run=False)
        acks = sum(1 for l in plan if l.get("status") not in (None, "UNCONFIRMED", "DRY-RUN", "SKIPPED-NDF"))
        log(f"FX sleeve TRANSMITTED: {acks}/{len([l for l in plan if l['tradeable']])} legs acknowledged.")
    finally:
        try: app.disconnect()
        except Exception: pass


if __name__ == "__main__":
    main()
