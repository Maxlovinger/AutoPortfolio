"""Tests for the (D) regime-switching model."""
import numpy as np
import pandas as pd
import pytest
import regime
from regime import detect_regime, regime_factor_weights, _identify_stress


def test_weights_calm_endpoint():
    w = regime_factor_weights({"p_stress": 0.0})
    assert w["momentum"] == pytest.approx(1.3)
    assert w["quality"] == pytest.approx(0.8)


def test_weights_stress_endpoint():
    w = regime_factor_weights({"p_stress": 1.0})
    assert w["quality"] == pytest.approx(1.5)
    assert w["momentum"] == pytest.approx(0.4)


def test_weights_midpoint_interpolates():
    w = regime_factor_weights({"p_stress": 0.5})
    assert w["momentum"] == pytest.approx((1.3 + 0.4) / 2)


def test_weights_clip_out_of_range():
    w = regime_factor_weights({"p_stress": 5.0})     # clipped to 1
    assert w["quality"] == pytest.approx(1.5)


def test_identify_stress_picks_high_variance():
    ret = pd.Series(np.concatenate([np.full(50, 0.001), np.full(50, 0.0)]))
    states = np.array([0] * 50 + [1] * 50)
    # state 0 has zero variance (constant), state 1 also constant -> both 0;
    # make state 1 volatile:
    ret = pd.Series(np.concatenate([np.full(50, 0.001),
                                    np.tile([0.05, -0.05], 25)]))
    assert _identify_stress(ret, states) == 1


def test_detect_regime_structure(calm_returns):
    r = detect_regime(returns=calm_returns)
    assert set(r) >= {"regime", "p_stress", "p_calm", "method"}
    assert 0.0 <= r["p_stress"] <= 1.0
    assert r["regime"] in {"calm", "stress"}
    assert r["p_calm"] == pytest.approx(1 - r["p_stress"])


def test_detect_regime_flags_stress_period(stress_returns):
    # series ends in a high-vol crash regime -> should not report pure calm=1.0
    r = detect_regime(returns=stress_returns)
    assert r["p_stress"] > 0.5


def test_short_series_returns_calm_none():
    short = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100))
    r = detect_regime(returns=short)
    assert r["method"] == "none"
    assert r["regime"] == "calm"


def test_detect_regime_offline(monkeypatch, stress_returns):
    # ensure the network path is not hit when returns supplied
    def boom(*a, **k):
        raise AssertionError("network should not be called")
    monkeypatch.setattr(regime.yf, "download", boom)
    r = detect_regime(returns=stress_returns)
    assert r["regime"] in {"calm", "stress"}
