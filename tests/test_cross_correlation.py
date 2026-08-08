"""Offline tests for cross_correlation.py pure helpers (no data files/plots)."""
import numpy as np
import pandas as pd
import pytest

import cross_correlation as cc


def test_to_monthly_compounds():
    idx = pd.bdate_range("2020-01-01", periods=63)      # ~3 months
    r = pd.Series(0.001, index=idx)
    m = cc._to_monthly(r)
    # each month's compounded return ~ (1.001^n - 1) > 0, one value per month
    assert (m > 0).all()
    assert 2 <= len(m) <= 4


def test_stats_known_series():
    r = pd.Series([0.01] * 24)                          # steady positive
    s = cc.stats(r)
    assert s["sharpe"] > 0
    assert s["maxdd"] == pytest.approx(0.0, abs=1e-9)   # never falls
    assert s["vol"] == pytest.approx(0.0, abs=1e-9)


def test_scale_to_vol_hits_target():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.005, 0.04, 240))
    scaled = cc._scale_to_vol(r, target=0.10)
    ann_vol = scaled.std(ddof=1) * np.sqrt(cc.PPY)
    assert ann_vol == pytest.approx(0.10, rel=1e-6)


def test_scale_to_vol_is_sharpe_invariant():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.005, 0.03, 240))
    assert cc.stats(r)["sharpe"] == pytest.approx(
        cc.stats(cc._scale_to_vol(r, 0.10))["sharpe"], rel=1e-9)
