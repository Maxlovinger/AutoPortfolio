"""
Edge-case tests for the walk-forward backtester. Fully offline: uses synthetic
prices and trivial deterministic strategies so behavior is exactly checkable,
plus point-in-time (no look-ahead) verification.
"""
import numpy as np
import pandas as pd
import pytest

import backtester as bt
from backtester import (
    walk_forward, benchmark_equal_weight, performance,
    weight_equal, weight_max_sharpe, score_momentum,
)
from tests.conftest import make_prices

UNIV = ["AAA", "BBB", "CCC", "DDD", "EEE"]


@pytest.fixture
def prices_bt():
    # enough history for lookback=120 + several rebalances
    return make_prices(UNIV, n_days=500, seed=11)


def test_performance_known_series():
    # constant +0.1%/day for 252 days
    r = pd.Series([0.001] * 252)
    m = performance(r, rf=0.0)
    assert m["cagr"] > 0
    assert m["max_dd"] == pytest.approx(0.0, abs=1e-9)   # never falls
    assert m["vol"] == pytest.approx(0.0, abs=1e-9)


def test_performance_drawdown_detected():
    r = pd.Series([0.05] * 10 + [-0.05] * 10)
    m = performance(r, rf=0.0)
    assert m["max_dd"] < 0


def test_performance_too_short_is_nan():
    m = performance(pd.Series([0.01]))
    assert np.isnan(m["sharpe"])


def test_walk_forward_runs_and_shapes(prices_bt):
    res = walk_forward(prices_bt, score_momentum, weight_equal,
                       top_n=3, lookback=120, rebalance=21, cost_bps=10)
    assert len(res["equity"]) > 0
    assert (res["equity"] > 0).all()
    # weights each rebalance sum to ~1 (rows that were actually set)
    row_sums = res["weights"].sum(axis=1)
    assert np.allclose(row_sums[row_sums > 0], 1.0, atol=1e-6)


def test_weights_respect_top_n(prices_bt):
    res = walk_forward(prices_bt, score_momentum, weight_equal,
                       top_n=2, lookback=120, rebalance=21)
    nonzero_per_row = (res["weights"] > 0).sum(axis=1)
    assert (nonzero_per_row[nonzero_per_row > 0] <= 2).all()


def test_no_lookahead_bias(prices_bt):
    """
    Truncating the price history must NOT change the equity path over the
    overlapping period — proves decisions use only past data.
    """
    full = walk_forward(prices_bt, score_momentum, weight_equal,
                        top_n=3, lookback=120, rebalance=21, cost_bps=0)
    cut = prices_bt.iloc[:400]
    part = walk_forward(cut, score_momentum, weight_equal,
                        top_n=3, lookback=120, rebalance=21, cost_bps=0)
    common = part["returns"].index.intersection(full["returns"].index)
    common = common[:-25]   # drop the tail where rebalance windows differ
    assert np.allclose(full["returns"].reindex(common),
                       part["returns"].reindex(common), atol=1e-9)


def test_costs_reduce_returns(prices_bt):
    free = walk_forward(prices_bt, score_momentum, weight_equal,
                        top_n=3, lookback=120, rebalance=21, cost_bps=0)
    costly = walk_forward(prices_bt, score_momentum, weight_equal,
                          top_n=3, lookback=120, rebalance=21, cost_bps=50)
    assert costly["equity"].iloc[-1] < free["equity"].iloc[-1]


def test_select_fn_overrides_top_n(prices_bt):
    """A custom select_fn should control which names are held each rebalance."""
    only = ["AAA", "BBB"]
    res = walk_forward(prices_bt, score_momentum, weight_equal,
                       top_n=5, lookback=120, rebalance=21,
                       select_fn=lambda s: only)
    held = [c for c in res["weights"].columns
            if (res["weights"][c] > 0).any()]
    assert set(held) <= set(only)


def test_insufficient_history_raises():
    small = make_prices(UNIV, n_days=80)
    with pytest.raises(ValueError):
        walk_forward(small, score_momentum, weight_equal, lookback=504)


def test_benchmark_equal_weight(prices_bt):
    eq, ret = benchmark_equal_weight(prices_bt, start_from=120)
    assert len(eq) > 0 and (eq > 0).all()


def test_weight_max_sharpe_capped(prices_bt):
    picks = ["AAA", "BBB", "CCC"]
    w = weight_max_sharpe(prices_bt, picks, cap=0.5, lookback=200)
    assert abs(w.sum() - 1) < 1e-6
    assert (w <= 0.5 + 1e-6).all()
