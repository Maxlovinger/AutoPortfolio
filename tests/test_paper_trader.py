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


# ---------------------------------------------------------------- fractional
def test_plan_orders_fractional_hits_exact_shares():
    # fractional target = nav*w/px exactly (no whole-share floor)
    w = pd.Series({"AAA": 0.5, "BBB": 0.5})
    prices = pd.Series({"AAA": 300.0, "BBB": 70.0})
    trades = plan_orders(w, prices, nav=43_000, current={}, fractional=True)
    by = {t["ticker"]: t for t in trades}
    assert by["AAA"]["shares"] == pytest.approx(21_500 / 300.0, abs=1e-4)   # 71.6667
    assert by["BBB"]["shares"] == pytest.approx(21_500 / 70.0, abs=1e-4)    # 307.1429
    # both are fractional (not whole)
    assert by["AAA"]["shares"] % 1 != 0 and by["BBB"]["shares"] % 1 != 0


def test_plan_orders_fractional_deploys_more_than_whole_share():
    # small account + pricey names (each budget buys ~1 whole share) => big cash
    # drag under whole-share; fractional deploys ~all of NAV.
    w = pd.Series({t: 1 / 5 for t in ["A", "B", "C", "D", "E"]})
    prices = pd.Series({"A": 5000.0, "B": 6000.0, "C": 7000.0, "D": 5500.0, "E": 6500.0})
    nav = 40_000.0                                     # $8k budget per name
    frac = plan_orders(w, prices, nav, current={}, fractional=True)
    whole = plan_orders(w, prices, nav, current={}, fractional=False)
    frac_deployed = sum(t["shares"] * prices[t["ticker"]] for t in frac)
    whole_deployed = sum(t["shares"] * prices[t["ticker"]] for t in whole)
    assert frac_deployed == pytest.approx(nav, abs=1.0)      # ~100% invested
    assert whole_deployed < nav * 0.80                        # ~25% stranded in cash
    assert frac_deployed > whole_deployed


def test_plan_orders_fractional_skips_dust():
    # a delta worth < min_notional is skipped as dust
    w = pd.Series({"AAA": 1.0})
    prices = pd.Series({"AAA": 100.0})
    # current already at 100.0000 shares; target 100.005 -> $0.50 dust < $1
    trades = plan_orders(w, prices, nav=10_000.5, current={"AAA": 100.0},
                         fractional=True, min_notional=1.0)
    assert trades == []


def test_plan_orders_fractional_trades_both_directions():
    w = pd.Series({"AAA": 0.5, "BBB": 0.5})
    prices = pd.Series({"AAA": 100.0, "BBB": 50.0})
    # AAA under target, BBB over target -> one BUY one SELL, fractional current
    trades = plan_orders(w, prices, nav=10_000, current={"AAA": 10.0, "BBB": 150.0},
                         fractional=True)
    by = {t["ticker"]: t for t in trades}
    assert by["AAA"]["action"] == "BUY"
    assert by["AAA"]["shares"] == pytest.approx(50.0 - 10.0, abs=1e-4)   # 5000/100 - 10
    assert by["BBB"]["action"] == "SELL"
    assert by["BBB"]["shares"] == pytest.approx(100.0 - 150.0, abs=1e-4)  # 5000/50 - 150


def test_plan_orders_whole_share_default_still_floors():
    # default (fractional=False) must be unchanged: integer floor
    w = pd.Series({"AAA": 1.0})
    prices = pd.Series({"AAA": 700.0})
    trades = plan_orders(w, prices, nav=10_000, current={})
    assert trades[0]["shares"] == 14 and isinstance(trades[0]["shares"], int)  # 10000//700


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
