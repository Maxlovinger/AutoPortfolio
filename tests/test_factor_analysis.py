"""
Tests for IC / quantile analysis. The key tests engineer factors with KNOWN
predictive power so we can verify the IC math points the right way.
"""
import numpy as np
import pandas as pd
import pytest

from factor_analysis import (
    forward_returns, factor_panels, ic_series, ic_summary,
    quantile_returns, is_monotonic, analyze,
)
from tests.conftest import make_prices

UNIV = [f"S{i}" for i in range(10)]


@pytest.fixture
def prices_fa():
    return make_prices(UNIV, n_days=800, seed=3)


def _panel(index, cols, values_fn):
    return pd.DataFrame({c: values_fn(i, c) for i, c in enumerate(cols)},
                        index=index)


def test_forward_returns_shape(prices_fa):
    fwd = forward_returns(prices_fa, horizon=21)
    assert fwd.shape == prices_fa.shape
    assert fwd.iloc[-21:].isna().all().all()   # last horizon rows have no future


def test_factor_panels_all_pit(prices_fa):
    panels = factor_panels(prices_fa)
    assert "momentum_12_1" in panels
    for name, p in panels.items():
        assert p.shape == prices_fa.shape


def test_perfect_factor_has_high_ic(prices_fa):
    """A factor equal to the forward return must have IC ~ +1."""
    fwd = forward_returns(prices_fa, horizon=21)
    ic = ic_series(fwd.copy(), fwd, sample=21)   # factor == future return
    summ = ic_summary(ic, horizon=21)
    assert summ["mean_ic"] > 0.95
    assert summ["hit_rate"] == pytest.approx(1.0)


def test_negated_factor_has_negative_ic(prices_fa):
    fwd = forward_returns(prices_fa, horizon=21)
    ic = ic_series(-fwd, fwd, sample=21)
    assert ic_summary(ic)["mean_ic"] < -0.95


def test_random_factor_has_near_zero_ic(prices_fa):
    fwd = forward_returns(prices_fa, horizon=21)
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(rng.normal(size=fwd.shape),
                         index=fwd.index, columns=fwd.columns)
    summ = ic_summary(ic_series(noise, fwd, sample=21))
    assert abs(summ["mean_ic"]) < 0.15          # no real predictive power


def test_perfect_factor_quantiles_monotonic(prices_fa):
    fwd = forward_returns(prices_fa, horizon=21)
    qr = quantile_returns(fwd.copy(), fwd, n_q=5, sample=21)
    assert is_monotonic(qr, 5)
    assert qr["long_short"] > 0                  # top beats bottom


def test_ic_summary_too_short_is_nan():
    summ = ic_summary(pd.Series([0.1]))
    assert np.isnan(summ["ic_ir"])


def test_ic_series_skips_thin_cross_sections():
    idx = pd.bdate_range("2020-01-01", periods=100)
    factor = pd.DataFrame({"A": range(100), "B": range(100)}, index=idx)
    fwd = factor.copy()
    ic = ic_series(factor, fwd, sample=21, min_names=5)  # only 2 names < 5
    assert len(ic) == 0


def test_quantile_all_nan_factor():
    idx = pd.bdate_range("2020-01-01", periods=100)
    factor = pd.DataFrame(np.nan, index=idx, columns=UNIV)
    fwd = pd.DataFrame(0.01, index=idx, columns=UNIV)
    qr = quantile_returns(factor, fwd, n_q=5, sample=21)
    assert qr[[c for c in qr.index if c.startswith("Q")]].isna().all()


def test_analyze_returns_summary(prices_fa):
    summary, quantiles, ics = analyze(prices_fa, horizon=21)
    assert not summary.empty
    assert set(["mean_ic", "ic_ir", "t_stat", "hit_rate", "monotonic"]) <= set(summary.columns)
    assert set(quantiles) == set(factor_panels(prices_fa))
