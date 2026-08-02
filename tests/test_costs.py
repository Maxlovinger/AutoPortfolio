"""
Tests for the realistic transaction-cost model. Fully offline / deterministic.
"""
import numpy as np
import pandas as pd
import pytest

from costs import (half_spread_bps, impact_bps, trade_cost_fraction,
                   rebalance_cost)


def test_half_spread_decreases_with_liquidity():
    thin = half_spread_bps(1e6)     # $1M ADV
    mid = half_spread_bps(1e8)      # $100M
    thick = half_spread_bps(1e9)    # $1B
    assert thin > mid > thick
    assert thin == pytest.approx(25.0, rel=1e-6)   # calibration anchor


def test_half_spread_clipped():
    # absurdly liquid floored, absurdly thin capped
    assert half_spread_bps(1e15) == pytest.approx(1.0)
    assert half_spread_bps(1.0) == pytest.approx(75.0)


def test_impact_monotonic_and_sqrt():
    assert impact_bps(0.0) == 0.0
    assert impact_bps(0.04) == pytest.approx(2 * impact_bps(0.01))  # sqrt law
    assert impact_bps(0.1) > impact_bps(0.01)


def test_trade_cost_fraction_positive_and_sized():
    # small trade in a liquid name is cheap; big trade in a thin name is dear
    cheap = trade_cost_fraction(1_000, 1e9)
    dear = trade_cost_fraction(1_000_000, 1e6)
    assert 0 < cheap < dear
    # cheap should be a handful of bps, dear should be large (>100bps)
    assert cheap < 10 / 1e4
    assert dear > 100 / 1e4


def test_rebalance_cost_zero_when_no_change():
    dw = pd.Series({"A": 0.0, "B": 0.0})
    adv = pd.Series({"A": 1e8, "B": 1e8})
    assert rebalance_cost(dw, adv, 1_000_000) == 0.0


def test_rebalance_cost_scales_with_capital():
    dw = pd.Series({"A": 0.5, "B": -0.5})
    adv = pd.Series({"A": 1e6, "B": 1e6})
    small = rebalance_cost(dw, adv, 100_000)
    big = rebalance_cost(dw, adv, 100_000_000)
    # bigger book -> more impact -> higher fractional cost
    assert big > small > 0


def test_rebalance_cost_missing_adv_uses_median():
    dw = pd.Series({"A": 1.0})
    adv = pd.Series({"B": 1e8, "C": 1e8})   # A missing
    # should not raise, should be finite and positive
    c = rebalance_cost(dw, adv, 1_000_000)
    assert np.isfinite(c) and c > 0


def test_realistic_costs_hurt_thin_names_more():
    """Same turnover, thinner ADV -> strictly higher cost."""
    dw = pd.Series({"A": 0.5, "B": -0.5})
    thick = pd.Series({"A": 1e9, "B": 1e9})
    thin = pd.Series({"A": 1e6, "B": 1e6})
    assert rebalance_cost(dw, thin, 1_000_000) > rebalance_cost(dw, thick, 1_000_000)
