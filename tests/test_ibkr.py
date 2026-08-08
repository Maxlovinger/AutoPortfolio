"""Unit tests for the ibkr.py order/contract builders (pure, no TWS connection)."""
import ibkr


# ---------------------------------------------------------------- contracts
def test_stock_dash_becomes_space():
    # IBKR uses a SPACE for share classes; yfinance/us use a DASH
    c = ibkr.stock("BRK-B")
    assert c.symbol == "BRK B" and c.secType == "STK"
    assert c.exchange == "SMART" and c.currency == "USD"


def test_stock_plain_symbol_unchanged():
    assert ibkr.stock("AAPL").symbol == "AAPL"


# ---------------------------------------------------------------- market orders
def test_market_order_fields():
    o = ibkr.market_order("BUY", 10)
    assert o.action == "BUY" and o.orderType == "MKT" and o.totalQuantity == 10
    # these must be forced False or TWS rejects the order
    assert getattr(o, "eTradeOnly", False) is False
    assert getattr(o, "firmQuoteOnly", False) is False


def test_market_on_open_is_opg():
    o = ibkr.market_on_open_order("SELL", 5)
    assert o.orderType == "MKT" and o.tif == "OPG" and o.action == "SELL"


# ---------------------------------------------------------------- fractional (cashQty)
def test_fractional_uses_cash_qty_not_fractional_shares():
    # IBKR rejects fractional SHARE quantities over the API (err 10243); the only
    # accepted route is a dollar-denominated cashQty order.
    o = ibkr.fractional_market_order("BUY", 549.37)
    assert o.orderType == "MKT" and o.tif == "DAY"
    assert o.cashQty == 549.37
    # must NOT smuggle a fractional share quantity (that is what gets rejected)
    assert not float(o.totalQuantity or 0) % 1     # 0 or whole, never fractional
    assert getattr(o, "eTradeOnly", False) is False


def test_fractional_rounds_cash_to_cents():
    o = ibkr.fractional_market_order("SELL", 12.3456)
    assert o.cashQty == 12.35 and o.action == "SELL"


def test_fractional_not_opg():
    # fractional orders cannot be Market-on-Open
    assert ibkr.fractional_market_order("BUY", 100.0).tif != "OPG"
