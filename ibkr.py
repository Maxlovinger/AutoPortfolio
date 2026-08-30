"""
ibkr.py — thin, synchronous wrapper over IB's official `ibapi` for TWS/Gateway.

Why ibapi (not ib_async): on this macOS box the `eventkit` dependency of
ib_async collides with Apple's pyobjc `EventKit` framework on the case-
insensitive filesystem and won't import. `ibapi` is pure-Python with no such
dependency.

ibapi is callback/threaded: we run the client message loop on a background
thread and use Events to turn async callbacks into blocking calls
(connect / account_summary / positions / place order).

Paper ports: TWS = 7497, IB Gateway = 4002.
Enable in TWS: Configure -> API -> Settings ->
  [x] Enable ActiveX and Socket Clients, Socket port 7497,
  add 127.0.0.1 to Trusted IPs (or check "Allow connections from localhost").
"""
from __future__ import annotations
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order

# benign "connection ok / market-data farm" notices, not real errors
INFO_CODES = {2104, 2106, 2107, 2108, 2119, 2158, 2100, 2150}

# HARD connectivity-loss codes: the socket to IB's servers is down, so any
# account/position read is stale-or-empty and any order we route may be dropped
# or filled against a phantom picture. A run that sees one of these MUST NOT
# transmit. (1100 = connectivity lost; 2110 = TWS<->server broken. Both are the
# ones that appeared in the field when the scheduled rebalance read NAV $0.)
CONNECTION_LOST_CODES = {1100, 2110}


def is_connection_lost(codes) -> bool:
    """True if any hard connectivity-loss code is present. `codes` may be an
    iterable of raw int codes or of (level, code, text) message tuples."""
    for c in codes:
        code = c[1] if isinstance(c, (tuple, list)) else c
        if code in CONNECTION_LOST_CODES:
            return True
    return False


class IBKRClient(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self._next_id = None
        self._id_evt = threading.Event()
        self.acct = {}
        self._acct_evt = threading.Event()
        self.positions = {}
        self._pos_evt = threading.Event()
        self.order_status = {}
        self.messages = []          # (level, code, text)

    # --- connection handshake ---
    def nextValidId(self, orderId: int):
        self._next_id = orderId
        self._id_evt.set()

    def error(self, reqId, errorCode="", errorString="", *args):
        level = "info" if errorCode in INFO_CODES else "error"
        self.messages.append((level, errorCode, errorString))

    # --- account summary ---
    def accountSummary(self, reqId, account, tag, value, currency):
        self.acct[tag] = value

    def accountSummaryEnd(self, reqId):
        self._acct_evt.set()

    # --- positions ---
    def position(self, account, contract, position, avgCost):
        self.positions[contract.symbol] = position

    def positionEnd(self):
        self._pos_evt.set()

    # --- orders ---
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, *a):
        self.order_status[orderId] = {"status": status, "filled": filled,
                                      "remaining": remaining,
                                      "avgFillPrice": avgFillPrice}

    def next_order_id(self) -> int:
        oid = self._next_id
        self._next_id += 1
        return oid


# ---------------------------------------------------------------- helpers
def stock(symbol: str) -> Contract:
    c = Contract()
    # IBKR uses a SPACE for share classes (BRK B, BF B), not the dash yfinance
    # uses (BRK-B). Without this, class-share tickers fail security lookup.
    c.symbol = symbol.replace("-", " ")
    c.secType = "STK"
    c.exchange = "SMART"
    c.currency = "USD"
    return c


def forex(base: str, quote: str = "USD") -> Contract:
    """Spot FX contract (IDEALPRO) for the carry sleeve, e.g. forex('EUR') = EUR.USD.
    Order quantity is in BASE-currency units (not shares). Provided for the FX
    sleeve; live FX execution is gated behind explicit opt-in (see portfolio_live)."""
    c = Contract()
    c.symbol = base
    c.secType = "CASH"
    c.currency = quote
    c.exchange = "IDEALPRO"
    return c


def market_order(action: str, quantity: int, tif: str = "DAY") -> Order:
    """MKT order. tif='DAY' (default, expires at close) or 'GTC' (Good-Till-Cancelled,
    persists across sessions until filled/cancelled). FX odd-lots use GTC so an
    unfilled leg keeps working the next session instead of expiring overnight."""
    o = Order()
    o.action = action                 # "BUY" / "SELL"
    o.orderType = "MKT"
    o.totalQuantity = quantity
    o.tif = tif
    # these default True in some ibapi builds and get rejected by TWS
    for attr in ("eTradeOnly", "firmQuoteOnly"):
        if hasattr(o, attr):
            setattr(o, attr, False)
    return o


def market_on_open_order(action: str, quantity: int) -> Order:
    """Market-on-Open: fills in the next opening auction. Can be staged while the
    market is closed (unlike a plain MKT order, which gets cancelled off-hours)."""
    o = market_order(action, quantity)
    o.tif = "OPG"                     # 'at the opening'
    return o


def fractional_market_order(action: str, cash_amount: float) -> Order:
    """Dollar-denominated MKT order (buys/sells $cash_amount of the stock, filling
    fractional shares). This is the ONLY fractional method IBKR accepts over the
    API: a fractional *share quantity* is rejected with error 10243 ("use desktop
    version"), so we specify a cash quantity (`cashQty`) instead.

    REQUIREMENTS (else IBKR returns 10244 "Cash Quantity cannot be used"):
      * the account must have US fractional-share trading permission enabled
        (Client Portal > Settings > Trading Permissions / Trading Experience),
      * regular trading hours, DAY tif (no Market-on-Open).
    Lets a small account sit at exact equal weight with no whole-share cash drag."""
    o = Order()
    o.action = action
    o.orderType = "MKT"
    o.tif = "DAY"
    o.cashQty = round(float(cash_amount), 2)   # dollar amount; IB fills fractional
    for attr in ("eTradeOnly", "firmQuoteOnly"):
        if hasattr(o, attr):
            setattr(o, attr, False)
    return o


def connect_tws(host="127.0.0.1", port=7497, client_id=17, timeout=12) -> IBKRClient:
    """Connect and block until TWS sends the nextValidId handshake."""
    app = IBKRClient()
    app.connect(host, port, client_id)
    th = threading.Thread(target=app.run, daemon=True)
    th.start()
    if not app._id_evt.wait(timeout):
        app.disconnect()
        raise ConnectionError(
            f"No handshake from TWS at {host}:{port}. In TWS enable "
            f"Configure > API > Settings: 'Enable ActiveX and Socket Clients', "
            f"Socket port {port}, and allow 127.0.0.1.")
    app._thread = th
    return app


def account_summary(app: IBKRClient, timeout=10) -> dict:
    app._acct_evt.clear()
    app.acct = {}
    rid = 9001
    app.reqAccountSummary(rid, "All",
                          "NetLiquidation,TotalCashValue,AvailableFunds,BuyingPower")
    app._acct_evt.wait(timeout)
    app.cancelAccountSummary(rid)
    return dict(app.acct)


def positions(app: IBKRClient, timeout=10) -> dict:
    app._pos_evt.clear()
    app.positions = {}
    app.reqPositions()
    app._pos_evt.wait(timeout)
    app.cancelPositions()
    return dict(app.positions)


def connection_test(host="127.0.0.1", port=7497, client_id=17):
    """Read-only connectivity check: connect, print account + positions, disconnect."""
    app = connect_tws(host, port, client_id)
    try:
        summ = account_summary(app)
        pos = positions(app)
        print(f"Connected to TWS at {host}:{port}")
        print(f"  NetLiquidation : {summ.get('NetLiquidation', 'n/a')}")
        print(f"  TotalCash      : {summ.get('TotalCashValue', 'n/a')}")
        print(f"  BuyingPower    : {summ.get('BuyingPower', 'n/a')}")
        print(f"  Open positions : {len(pos)}  {dict(list(pos.items())[:8])}")
        errs = [m for m in app.messages if m[0] == "error"]
        if errs:
            print(f"  Notices/errors : {errs[:5]}")
        return {"account": summ, "positions": pos}
    finally:
        app.disconnect()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7497
    connection_test(port=port)
