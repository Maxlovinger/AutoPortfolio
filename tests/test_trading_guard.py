"""
Tests for the pre-trade safety guards that stop the live jobs from routing
orders against an untrusted IB Gateway feed.

Regression target: 2026-08-24, the scheduled rebalance ran while the gateway
connection was broken, account_summary() timed out, nav() returned its 0.0
default, and the run logged "NAV $0". It happened to send 0 orders only because
positions ALSO came back empty. Had positions loaded with nav=0, plan_orders
would have sized every target to 0 shares -> SELL-the-entire-book, transmitted
live. These guards make that abort instead.
"""
import numpy as np
import pytest

from ibkr import is_connection_lost, CONNECTION_LOST_CODES
from paper_trader import check_tradeable, TradingHalt, plan_orders


# --------------------------------------------------------------------------
# is_connection_lost — accepts raw codes or (level, code, text) message tuples
# --------------------------------------------------------------------------
def test_connection_lost_raw_codes():
    assert is_connection_lost([2110]) is True
    assert is_connection_lost([1100]) is True
    assert is_connection_lost([2104]) is False        # info: farm ok
    assert is_connection_lost([]) is False


def test_connection_lost_message_tuples():
    assert is_connection_lost([("error", 2110, "broken")]) is True
    assert is_connection_lost([("info", 2104, "farm ok")]) is False
    # mixed stream: one hard loss among benign notices still trips
    assert is_connection_lost(
        [("info", 2104, "ok"), ("error", 1100, "lost")]) is True


def test_connection_lost_codes_are_the_field_codes():
    # the exact codes that appeared in auto_run.log / fx_run.log on the outage
    assert 2110 in CONNECTION_LOST_CODES and 1100 in CONNECTION_LOST_CODES


# --------------------------------------------------------------------------
# check_tradeable — the gate before plan_orders in auto_rebalance
# --------------------------------------------------------------------------
GOOD_POS = {"AAPL": 10, "IEF": 254}
HELD = ["AAPL", "IEF"]


def test_healthy_feed_passes():
    # positive NAV, positions match what we expect to hold, no error codes
    check_tradeable(250_000.0, GOOD_POS, HELD, error_codes=[("info", 2104, "ok")])


def test_zero_nav_halts():
    with pytest.raises(TradingHalt, match="NAV unreadable"):
        check_tradeable(0.0, GOOD_POS, HELD)


def test_negative_nav_halts():
    with pytest.raises(TradingHalt):
        check_tradeable(-5.0, GOOD_POS, HELD)


def test_none_nav_halts():
    with pytest.raises(TradingHalt):
        check_tradeable(None, GOOD_POS, HELD)


def test_nan_nav_halts():
    with pytest.raises(TradingHalt):
        check_tradeable(float("nan"), GOOD_POS, HELD)


def test_connectivity_code_halts_even_with_good_nav():
    # a stale-but-positive NAV must not be trusted if the socket dropped
    with pytest.raises(TradingHalt, match="connectivity lost"):
        check_tradeable(250_000.0, GOOD_POS, HELD,
                        error_codes=[("error", 2110, "broken")])


def test_phantom_empty_positions_halts():
    # we believe we hold a book but the positions feed returned nothing
    with pytest.raises(TradingHalt, match="phantom-empty"):
        check_tradeable(250_000.0, {}, HELD)


def test_all_zero_positions_halts():
    # positions present but all quantities zero -> still a phantom-empty book
    with pytest.raises(TradingHalt, match="phantom-empty"):
        check_tradeable(250_000.0, {"AAPL": 0, "IEF": 0}, HELD)


def test_first_run_with_no_holdings_is_allowed():
    # inception: we hold nothing yet, so an empty positions read is CORRECT and
    # the run must be allowed to buy the book fresh
    check_tradeable(250_000.0, {}, expected_holdings=[])


def test_info_code_does_not_halt():
    check_tradeable(250_000.0, GOOD_POS, HELD,
                    error_codes=[("info", 2104, "farm ok"), ("info", 2106, "hmds")])


# --------------------------------------------------------------------------
# The catastrophe this prevents, shown concretely: nav=0 -> sell everything
# --------------------------------------------------------------------------
def test_nav_zero_would_liquidate_book_without_guard():
    import pandas as pd
    weights = pd.Series({"AAPL": 0.5, "IEF": 0.5})
    prices = pd.Series({"AAPL": 200.0, "IEF": 95.0})
    current = {"AAPL": 10, "IEF": 254}
    # with a phantom nav=0, plan_orders wants to SELL the entire book
    trades = plan_orders(weights, prices, nav=0.0, current=current)
    assert all(t["action"] == "SELL" for t in trades)
    assert {t["ticker"] for t in trades} == {"AAPL", "IEF"}
    # ...which is exactly what check_tradeable refuses to let happen
    with pytest.raises(TradingHalt):
        check_tradeable(0.0, current, list(current))


# --------------------------------------------------------------------------
# run_fx App flags connectivity loss so it can abort before transmitting
# --------------------------------------------------------------------------
def test_run_fx_app_flags_connectivity_loss(monkeypatch):
    import run_fx
    monkeypatch.setattr(run_fx, "log", lambda *a, **k: None)  # no log file writes
    app = run_fx.App()
    assert app.conn_lost is False
    app.error(-1, 2104, "market data farm ok")   # benign
    assert app.conn_lost is False
    app.error(-1, 2110, "Connectivity broken")    # hard loss
    assert app.conn_lost is True
