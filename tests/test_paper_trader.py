"""Offline tests for the paper-trading loop (SimBroker)."""
import json
import numpy as np
import pandas as pd
import pytest
from paper_trader import SimBroker, plan_orders


def test_plan_orders_buys_from_flat():
    w = pd.Series({"AAA": 0.5, "BBB": 0.5})
    prices = pd.Series({"AAA": 100.0, "BBB": 50.0})
    trades = plan_orders(w, prices, nav=100_000, current={})
    by = {t["ticker"]: t for t in trades}
    assert by["AAA"]["action"] == "BUY" and by["AAA"]["shares"] == 500   # 50k/100
    assert by["BBB"]["shares"] == 1000                                   # 50k/50


def test_plan_orders_sells_dropped_name():
    w = pd.Series({"AAA": 1.0})
    prices = pd.Series({"AAA": 100.0, "BBB": 50.0})
    trades = plan_orders(w, prices, nav=100_000, current={"AAA": 500, "BBB": 200})
    by = {t["ticker"]: t for t in trades}
    assert by["BBB"]["action"] == "SELL" and by["BBB"]["shares"] == -200


def test_plan_orders_no_trade_when_on_target():
    w = pd.Series({"AAA": 1.0})
    prices = pd.Series({"AAA": 100.0})
    trades = plan_orders(w, prices, nav=100_000, current={"AAA": 1000})
    assert trades == []


def test_plan_orders_skips_bad_price():
    w = pd.Series({"AAA": 0.5, "BAD": 0.5})
    prices = pd.Series({"AAA": 100.0, "BAD": np.nan})
    trades = plan_orders(w, prices, nav=100_000, current={})
    assert all(t["ticker"] != "BAD" for t in trades)


@pytest.fixture
def broker(tmp_path):
    return SimBroker(state_file=str(tmp_path / "state.json"),
                     start_cash=100_000, cost_bps=10)


def test_initial_nav_is_cash(broker):
    assert broker.nav(pd.Series(dtype=float)) == pytest.approx(100_000)


def test_rebalance_hits_target_weights(broker):
    prices = pd.Series({"AAA": 100.0, "BBB": 50.0})
    w = pd.Series({"AAA": 0.6, "BBB": 0.4})
    broker.rebalance(w, prices)
    nav = broker.nav(prices)
    aaa_val = broker.state["positions"]["AAA"] * 100
    assert aaa_val / nav == pytest.approx(0.6, abs=0.02)   # whole-share rounding
    assert nav <= 100_000                                  # costs reduce NAV


def test_costs_charged(broker):
    prices = pd.Series({"AAA": 100.0})
    broker.rebalance(pd.Series({"AAA": 1.0}), prices)
    assert broker.nav(prices) < 100_000                    # 10bps buy-in cost


def test_selling_to_zero_removes_position(broker):
    prices = pd.Series({"AAA": 100.0, "BBB": 50.0})
    broker.rebalance(pd.Series({"AAA": 0.5, "BBB": 0.5}), prices)
    broker.rebalance(pd.Series({"AAA": 1.0}), prices)      # drop BBB
    assert "BBB" not in broker.state["positions"]


def test_snapshot_and_persistence(broker, tmp_path):
    prices = pd.Series({"AAA": 100.0, "BBB": 50.0})
    broker.rebalance(pd.Series({"AAA": 0.5, "BBB": 0.5}), prices)
    broker.snapshot(prices)
    broker.save()
    reloaded = SimBroker(state_file=broker.state_file)
    assert len(reloaded.state["history"]) == 1
    assert reloaded.state["positions"] == broker.state["positions"]


def test_rebalance_skips_bad_prices(broker):
    prices = pd.Series({"AAA": 100.0, "BAD": np.nan})
    trades = broker.rebalance(pd.Series({"AAA": 0.5, "BAD": 0.5}), prices)
    assert all(t["ticker"] != "BAD" for t in trades)


def test_nav_conserved_ex_costs(broker):
    """With zero cost, rebalancing must conserve NAV (cash + positions)."""
    b = SimBroker(state_file=broker.state_file + "2", start_cash=100_000, cost_bps=0)
    prices = pd.Series({"AAA": 100.0, "BBB": 50.0, "CCC": 25.0})
    b.rebalance(pd.Series({"AAA": 0.4, "BBB": 0.3, "CCC": 0.3}), prices)
    assert b.nav(prices) == pytest.approx(100_000, abs=200)   # only share-rounding drift
