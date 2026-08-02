"""Correctness + edge cases for the Markowitz optimizer."""
import numpy as np
import pytest
from data import returns_stats
from markowitz import (
    portfolio_return, portfolio_vol, sharpe_ratio,
    min_variance, max_sharpe, efficient_return, efficient_frontier,
)


@pytest.fixture
def mu_sig(prices):
    mu, Sig, _ = returns_stats(prices)
    return mu, Sig


def test_weights_sum_to_one(mu_sig):
    mu, Sig = mu_sig
    for w in (min_variance(Sig), max_sharpe(mu, Sig, 0.04)):
        assert abs(w.sum() - 1.0) < 1e-6


def test_long_only_bounds(mu_sig):
    mu, Sig = mu_sig
    w = max_sharpe(mu, Sig, 0.04)
    assert (w >= -1e-6).all() and (w <= 1 + 1e-6).all()


def test_min_variance_is_lowest_vol(mu_sig):
    mu, Sig = mu_sig
    w_mv = min_variance(Sig)
    n = len(mu)
    equal = np.repeat(1 / n, n)
    assert portfolio_vol(w_mv, Sig) <= portfolio_vol(equal, Sig) + 1e-9


def test_max_sharpe_beats_equal_weight(mu_sig):
    mu, Sig = mu_sig
    n = len(mu)
    equal = np.repeat(1 / n, n)
    w_ms = max_sharpe(mu, Sig, 0.04)
    assert sharpe_ratio(w_ms, mu, Sig, 0.04) >= sharpe_ratio(equal, mu, Sig, 0.04) - 1e-9


def test_efficient_return_hits_target(mu_sig):
    mu, Sig = mu_sig
    target = float(np.mean(mu))
    w = efficient_return(mu, Sig, target)
    assert abs(portfolio_return(w, mu) - target) < 1e-4


def test_frontier_vol_nondecreasing(mu_sig):
    mu, Sig = mu_sig
    vols, rets, _ = efficient_frontier(mu, Sig, n_points=25)
    # by construction returns are increasing; vol should be ~convex/non-decreasing
    assert np.all(np.diff(vols) > -1e-3)


def test_single_asset():
    mu = np.array([0.1])
    Sig = np.array([[0.04]])
    w = min_variance(Sig)
    assert abs(w[0] - 1.0) < 1e-9
    assert abs(max_sharpe(mu, Sig, 0.0)[0] - 1.0) < 1e-9


def test_allow_short_can_go_negative():
    # two anti-correlated assets, target below both -> shorting needed sometimes
    mu = np.array([0.20, 0.05])
    Sig = np.array([[0.04, -0.01], [-0.01, 0.02]])
    w = efficient_return(mu, Sig, target=0.25, allow_short=True)
    assert abs(w.sum() - 1) < 1e-6
    assert w.min() < 0  # achieving 0.25 (> max mu 0.20) requires a short leg
