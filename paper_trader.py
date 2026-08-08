"""
paper_trader.py — forward paper-trading validation loop.

>>> DEPRECATED STRATEGY PATH <<<
    target_weights() / run_once() / main() here run the OLD research strategy
    (score_combined top-N max-Sharpe) and are kept only for reference. The LIVE
    system is now auto_rebalance.py, which runs the LOCKED strategy (liquid-30,
    equal-weight, quarterly, weekly-banded vol-target). This file is retained
    because auto_rebalance REUSES its still-current, tested infrastructure:
    SimBroker, IBKRBroker, and plan_orders(). Do not start new work from main().

WHY THIS EXISTS
---------------
A historical backtest on today's constituents suffers survivorship bias. Forward
paper trading does NOT: you pick from the LIVE universe in real time and record
what actually happens, so there is no way to accidentally exclude names that
hadn't failed yet. Run this on a schedule (e.g. monthly) and it accumulates an
honest, bias-free track record.

WHAT IT DOES each run:
  1. Loads the tradeable universe (universe_tickers.txt)
  2. Scores stocks with the SAME point-in-time strategy as the backtester
  3. Computes target weights (top-N, capped max-Sharpe)
  4. Rebalances the paper portfolio to those weights (charging costs)
  5. Marks to market at live prices and appends a snapshot to the track record

BROKERS
  * SimBroker  (default): a self-contained paper account persisted to JSON —
    works TODAY with no external setup, marks to market via yfinance.
  * IBKRBroker (optional): routes real paper orders to Interactive Brokers via
    IB's official ibapi (see ibkr.py). Requires TWS/IB Gateway running with a
    paper account. dry_run=True by default (previews orders without sending).
"""
from __future__ import annotations
import json
import os
import datetime as dt
import numpy as np
import pandas as pd

STATE_FILE = "paper_state.json"
TRACK_FILE = "paper_track_record.csv"
START_CASH = 100_000.0
TOP_N = 10
COST_BPS = 10.0


# ----------------------------------------------------------------------
# Strategy: target weights from the current (point-in-time = today) data
# ----------------------------------------------------------------------
def target_weights(prices: pd.DataFrame, top_n=TOP_N) -> pd.Series:
    """Reuse the backtester's price-based strategy to pick + weight stocks."""
    from backtester import score_combined, weight_max_sharpe
    scores = score_combined(prices, use_ml=False)
    picks = list(scores.sort_values(ascending=False).head(top_n).index)
    return weight_max_sharpe(prices, picks)


# ----------------------------------------------------------------------
# Self-contained paper broker (works today, no external deps)
# ----------------------------------------------------------------------
class SimBroker:
    def __init__(self, state_file=STATE_FILE, start_cash=START_CASH, cost_bps=COST_BPS):
        self.state_file = state_file
        self.cost_bps = cost_bps
        if os.path.exists(state_file):
            with open(state_file) as f:
                self.state = json.load(f)
        else:
            self.state = {"cash": start_cash, "positions": {},
                          "inception": dt.date.today().isoformat(), "history": []}

    def nav(self, prices: pd.Series) -> float:
        pos_val = sum(sh * float(prices.get(t, 0.0))
                      for t, sh in self.state["positions"].items())
        return self.state["cash"] + pos_val

    def rebalance(self, weights: pd.Series, prices: pd.Series):
        """Trade toward target weights; whole shares; charge bps cost per trade."""
        nav = self.nav(prices)
        target_shares = {}
        for t, w in weights.items():
            px = float(prices.get(t, np.nan))
            if not np.isfinite(px) or px <= 0:
                continue
            target_shares[t] = int((nav * w) // px)   # whole shares
        trades = []
        universe = set(self.state["positions"]) | set(target_shares)
        for t in universe:
            cur = self.state["positions"].get(t, 0)
            tgt = target_shares.get(t, 0)
            d = tgt - cur
            if d == 0:
                continue
            px = float(prices.get(t, np.nan))
            if not np.isfinite(px) or px <= 0:
                continue
            notional = d * px
            cost = abs(notional) * self.cost_bps / 1e4
            self.state["cash"] -= notional + cost
            if tgt == 0:
                self.state["positions"].pop(t, None)
            else:
                self.state["positions"][t] = tgt
            trades.append({"ticker": t, "shares": d, "price": round(px, 2)})
        return trades

    def snapshot(self, prices: pd.Series):
        nav = self.nav(prices)
        weights = {t: round(sh * float(prices.get(t, 0.0)) / nav, 4)
                   for t, sh in self.state["positions"].items()} if nav else {}
        self.state["history"].append({
            "date": dt.date.today().isoformat(), "nav": round(nav, 2),
            "cash": round(self.state["cash"], 2),
            "n_positions": len(self.state["positions"]), "weights": weights})
        return nav

    def save(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)
        # also append a flat track-record row for easy plotting
        h = self.state["history"]
        if h:
            pd.DataFrame(h)[["date", "nav", "cash", "n_positions"]] \
                .to_csv(TRACK_FILE, index=False)


# ----------------------------------------------------------------------
# Order planning (pure, testable) — shared target-share math
# ----------------------------------------------------------------------
def plan_orders(weights: pd.Series, prices: pd.Series, nav: float,
                current: dict, fractional: bool = False,
                min_notional: float = 1.0) -> list[dict]:
    """
    Given target weights, live prices, portfolio NAV and current share holdings,
    return the BUY/SELL trades needed to reach the target.
    Names dropped from the target are fully sold.

    fractional=False (default): whole-share targets (floor). Leaves rounding cash
    idle on small accounts, but works on any account.
    fractional=True: exact fractional-share targets (round to 4dp), so the book
    can sit at exact equal weight with ~no idle cash. Requires the broker/account
    to support fractional shares. Deltas whose notional is below `min_notional`
    are skipped as dust.
    """
    target_shares = {}
    for t, w in weights.items():
        px = float(prices.get(t, np.nan))
        if np.isfinite(px) and px > 0:
            target_shares[t] = round(nav * float(w) / px, 4) if fractional \
                else int((nav * float(w)) // px)
    trades = []
    for t in set(current) | set(target_shares):
        cur = float(current.get(t, 0)) if fractional else int(current.get(t, 0))
        tgt = target_shares.get(t, 0)
        d = round(tgt - cur, 4) if fractional else int(tgt) - cur
        if d == 0:
            continue
        px = float(prices.get(t, np.nan))
        if not np.isfinite(px) or px <= 0:
            continue
        if fractional and abs(d * px) < min_notional:
            continue                    # skip sub-dollar dust trades
        trades.append({"ticker": t, "shares": d, "action": "BUY" if d > 0 else "SELL",
                       "price": round(px, 2)})
    return trades


# ----------------------------------------------------------------------
# IBKR backend (real paper orders) — via IB's official ibapi (ibkr.py)
# ----------------------------------------------------------------------
class IBKRBroker:
    """
    Routes paper orders to Interactive Brokers TWS/Gateway using ibkr.py (ibapi).

    Requires TWS or IB Gateway running with a PAPER account and the API enabled
    (Configure > API > Settings). Paper ports: TWS=7497, IB Gateway=4002.

    SAFETY: dry_run=True (default) computes and logs intended orders but does NOT
    transmit them. Set dry_run=False to actually place paper orders.
    """
    def __init__(self, host="127.0.0.1", port=7497, client_id=17,
                 cost_bps=COST_BPS, dry_run=True):
        import ibkr
        self._ibkr = ibkr
        self.app = ibkr.connect_tws(host, port, client_id)
        self.dry_run = dry_run
        self.cost_bps = cost_bps

    def nav(self, prices=None) -> float:
        summ = self._ibkr.account_summary(self.app)
        return float(summ.get("NetLiquidation", 0.0))

    def rebalance(self, weights: pd.Series, prices: pd.Series):
        nav = self.nav()
        current = {k: int(v) for k, v in self._ibkr.positions(self.app).items()}
        trades = plan_orders(weights, prices, nav, current)
        for tr in trades:
            if not self.dry_run:
                oid = self.app.next_order_id()
                self.app.placeOrder(oid, self._ibkr.stock(tr["ticker"]),
                                    self._ibkr.market_order(tr["action"],
                                                            abs(tr["shares"])))
                tr["order_id"] = oid
        self._last_trades = trades
        return trades

    def snapshot(self, prices=None):
        nav = self.nav()
        self._snap = {"date": dt.date.today().isoformat(), "nav": round(nav, 2),
                      "n_positions": len(self._ibkr.positions(self.app))}
        return nav

    def save(self):
        """Append the NAV snapshot to a dedicated IBKR track record."""
        if not hasattr(self, "_snap"):
            return
        path = "ibkr_track_record.csv"
        row = pd.DataFrame([self._snap])
        if os.path.exists(path):
            row = pd.concat([pd.read_csv(path), row], ignore_index=True)
        row.to_csv(path, index=False)

    def disconnect(self):
        try:
            self.app.disconnect()
        except Exception:
            pass


# ----------------------------------------------------------------------
# The loop: one iteration
# ----------------------------------------------------------------------
def run_once(broker=None, universe_file="universe_tickers.txt", top_n=TOP_N,
             verbose=True):
    from data import download_prices
    from universe import load_universe

    broker = broker or SimBroker()
    universe = load_universe(universe_file)
    prices = download_prices(universe, start="2018-01-01")
    if prices is None or len(prices) == 0 or prices.shape[1] == 0:
        raise RuntimeError("No price data returned (yfinance rate limit or bad "
                           "tickers). Retry shortly.")
    latest = prices.iloc[-1]

    weights = target_weights(prices, top_n=top_n)
    trades = broker.rebalance(weights, latest)
    nav = broker.snapshot(latest)
    broker.save()

    if verbose:
        print(f"[{dt.date.today()}] rebalanced -> NAV ${nav:,.0f}, "
              f"{len(trades)} trades, {len(weights)} target holdings")
        print("Target weights:")
        for t, w in weights.sort_values(ascending=False).items():
            print(f"  {t:6s} {w*100:5.1f}%")
    return {"nav": nav, "trades": trades, "weights": weights}


def make_broker(kind="sim", dry_run=True, port=7497):
    """kind: 'sim' (self-contained) or 'ibkr' (route to TWS/Gateway)."""
    if kind == "ibkr":
        return IBKRBroker(port=port, dry_run=dry_run)
    return SimBroker()


def main():
    import sys
    kind = "ibkr" if "--ibkr" in sys.argv else "sim"
    dry_run = "--live" not in sys.argv          # transmit only with explicit --live
    port = 4002 if "--gateway" in sys.argv else 7497
    universe_file = "demo_tickers.txt" if "--demo" in sys.argv else "universe_tickers.txt"

    broker = make_broker(kind, dry_run=dry_run, port=port)
    if kind == "ibkr":
        mode = "LIVE (transmitting)" if not dry_run else "DRY-RUN (no orders sent)"
        print(f"IBKR mode: {mode}  port={port}")
    try:
        res = run_once(broker=broker, universe_file=universe_file)
        if kind == "ibkr":
            print("\nIntended orders:" if dry_run else "\nOrders sent:")
            for tr in res["trades"]:
                print(f"  {tr['action']:4s} {abs(tr['shares']):>5d} {tr['ticker']:6s} "
                      f"@ ~${tr['price']}")
    finally:
        if hasattr(broker, "disconnect"):
            broker.disconnect()

    if os.path.exists(TRACK_FILE):
        tr = pd.read_csv(TRACK_FILE)
        print(f"\nTrack record ({len(tr)} snapshots):")
        print(tr.tail(6).to_string(index=False))
    print("\nUsage: python3 paper_trader.py [--ibkr] [--live] [--gateway] [--demo]")
    print("  default = SimBroker.  --ibkr routes to TWS (dry-run).  add --live to transmit.")


if __name__ == "__main__":
    main()
