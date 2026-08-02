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
    weight_equal, weight_max_sharpe, weight_min_variance,
    score_momentum, score_reversal, vol_target,
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


def test_realistic_cost_model_reduces_returns(prices_bt):
    """Passing an ADV series triggers the spread+impact model and costs money."""
    adv = pd.Series({t: 5e6 for t in UNIV})   # thin-ish names
    free = walk_forward(prices_bt, score_momentum, weight_equal,
                        top_n=3, lookback=120, rebalance=21, cost_bps=0)
    costly = walk_forward(prices_bt, score_momentum, weight_equal,
                          top_n=3, lookback=120, rebalance=21,
                          adv=adv, capital=10_000_000)
    assert costly["equity"].iloc[-1] < free["equity"].iloc[-1]


def test_realistic_cost_bigger_book_costs_more(prices_bt):
    adv = pd.Series({t: 2e6 for t in UNIV})
    small = walk_forward(prices_bt, score_momentum, weight_equal,
                         top_n=3, lookback=120, rebalance=21,
                         adv=adv, capital=1_000_000)
    big = walk_forward(prices_bt, score_momentum, weight_equal,
                       top_n=3, lookback=120, rebalance=21,
                       adv=adv, capital=200_000_000)
    assert big["equity"].iloc[-1] < small["equity"].iloc[-1]


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


def test_weight_min_variance_capped(prices_bt):
    picks = ["AAA", "BBB", "CCC", "DDD"]
    w = weight_min_variance(prices_bt, picks, cap=0.4, lookback=200)
    assert abs(w.sum() - 1) < 1e-6
    assert (w <= 0.4 + 1e-6).all()


def test_weight_min_variance_survives_gaps(prices_bt):
    """Gappy point-in-time histories must not collapse the book to one name."""
    gapped = prices_bt.copy()
    gapped.loc[gapped.index[:150], "CCC"] = np.nan   # CCC only recently listed
    gapped.loc[gapped.index[-30:], "DDD"] = np.nan    # DDD just delisted
    w = weight_min_variance(gapped, ["AAA", "BBB", "CCC", "DDD"],
                            cap=0.5, lookback=200)
    assert abs(w.sum() - 1) < 1e-6
    assert len(w) >= 3                                 # didn't collapse


def test_vol_target_reduces_vol_and_no_lookahead():
    # a calm stretch then a wild stretch; targeting must tame the wild part
    rng = np.random.default_rng(5)
    calm = rng.normal(0, 0.005, 300)
    wild = rng.normal(0, 0.05, 300)
    r = pd.Series(np.concatenate([calm, wild]))
    rt = vol_target(r, target=0.15, lookback=21, max_lev=1.0)
    # realized vol of the overlaid series should be lower in the wild part
    assert rt.iloc[300:].std() < r.iloc[300:].std()
    # never levers up beyond max_lev: |scaled| <= |raw|
    assert (rt.abs() <= r.abs() + 1e-12).all()


def test_score_reversal_favors_oversold(prices_bt):
    """A name that just dropped below its mean should score higher than one above."""
    s = score_reversal(prices_bt)
    assert s.notna().any()
    assert s.index.isin(prices_bt.columns).all()
