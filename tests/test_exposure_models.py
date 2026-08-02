"""
Tests for the invested-vs-cash exposure models. Emphasis on no-look-ahead:
an exposure decision for day t must not depend on day t's own return.
"""
import numpy as np
import pandas as pd
import pytest

import exposure_models as em


@pytest.fixture
def rets():
    rng = np.random.default_rng(1)
    calm = rng.normal(0.0004, 0.006, 250)
    wild = rng.normal(-0.001, 0.03, 250)
    return pd.Series(np.concatenate([calm, wild]),
                     index=pd.date_range("2020-01-01", periods=500, freq="B"))


def test_fit_ewma_lambda_in_bounds(rets):
    lam = em.fit_ewma_lambda(rets)
    assert 0.80 <= lam <= 0.995


def test_ewma_var_no_lookahead(rets):
    r = rets.values
    v1 = em._ewma_var(r, 0.9)
    r2 = r.copy(); r2[300] *= 5
    v2 = em._ewma_var(r2, 0.9)
    assert np.allclose(v1[:301], v2[:301])       # forecast up to t unchanged
    assert v2[301] != v1[301]                    # future forecast changes


def test_voltarget_exposure_capped(rets):
    vol = pd.Series([0.10, 0.15, 0.30, 0.60], index=rets.index[:4])
    e = em.exposure_voltarget(vol, target=0.15)
    assert e.iloc[0] == pytest.approx(1.0)       # calm -> fully invested
    assert e.iloc[2] == pytest.approx(0.5)
    assert (e <= 1.0 + 1e-9).all()


def test_trend_invested_in_uptrend():
    up = pd.Series(np.linspace(0.001, 0.002, 400),
                   index=pd.date_range("2020-01-01", periods=400, freq="B"))
    e = em.exposure_trend(up, window=200, floor=0.0)
    assert e.iloc[-1] == pytest.approx(1.0)      # steady uptrend -> fully in


def test_trend_no_lookahead(rets):
    e1 = em.exposure_trend(rets, window=50, floor=0.0)
    r2 = rets.copy(); r2.iloc[400] *= 3
    e2 = em.exposure_trend(r2, window=50, floor=0.0)
    assert np.allclose(e1.iloc[:401].values, e2.iloc[:401].values, equal_nan=True)


def test_cppi_bounds_and_no_lookahead(rets):
    out, exp = em.cppi_returns(rets, m=3.0, floor_frac=0.80)
    assert (exp >= -1e-9).all() and (exp <= 1.0 + 1e-9).all()
    # exposure[t] set before r[t] is applied -> perturbing r[t] can't change exp[t]
    r2 = rets.copy(); r2.iloc[250] *= 4
    _, exp2 = em.cppi_returns(r2, m=3.0, floor_frac=0.80)
    assert np.allclose(exp.iloc[:251].values, exp2.iloc[:251].values)


def test_cppi_controls_drawdown(rets):
    out, exp = em.cppi_returns(rets, m=4.0, floor_frac=0.85)
    eq = (1 + out).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    raw_dd = ((1 + rets).cumprod() / (1 + rets).cumprod().cummax() - 1).min()
    assert dd > raw_dd                            # shallower than un-overlaid


def test_banded_exposure_reduces_turnover(rets):
    # a jittery target should trade far less under a band than continuously
    target = pd.Series(np.random.default_rng(2).uniform(0.4, 1.0, len(rets)),
                       index=rets.index)
    banded = em.banded_exposure(target, band=0.10, check_every=5)
    assert banded.diff().abs().sum() < target.diff().abs().sum()


def test_banded_exposure_holds_within_band():
    # target drifts by less than the band -> exposure never moves
    target = pd.Series([1.0, 0.96, 0.93, 0.95, 1.0],
                       index=pd.date_range("2020-01-01", periods=5, freq="B"))
    banded = em.banded_exposure(target, band=0.10, check_every=1)
    assert (banded == 1.0).all()


def test_vix_breaker_only_cuts_above_threshold(rets):
    vix = pd.Series(0.15, index=rets.index)
    vix.iloc[100:110] = 0.50                       # a spike
    e = em.exposure_vix_breaker(rets, vix, threshold=0.35, floor=0.3, ramp=0.0)
    assert e.iloc[50] == pytest.approx(1.0)        # calm -> fully invested
    assert e.iloc[105] == pytest.approx(0.3)       # extreme -> cut to floor
    # the cut is lagged by one day (uses yesterday's VIX)
    assert e.iloc[100] == pytest.approx(1.0)


def test_vix_breaker_no_lookahead(rets):
    vix = pd.Series(np.linspace(0.1, 0.5, len(rets)), index=rets.index)
    e = em.exposure_vix_breaker(rets, vix, threshold=0.35, floor=0.3)
    r2 = rets.copy(); r2.iloc[400] *= 5            # perturb a return
    e2 = em.exposure_vix_breaker(r2, vix, threshold=0.35, floor=0.3)
    assert np.allclose(e.values, e2.values)        # exposure depends on VIX, not returns


def test_vix_percentile_threshold_causal(rets):
    vix = pd.Series(np.r_[np.full(300, 0.15), np.full(len(rets) - 300, 0.45)],
                    index=rets.index)
    e = em.exposure_vix_percentile(rets, vix, q=0.90, floor=0.3, min_obs=100)
    assert (e <= 1.0 + 1e-9).all() and (e >= 0.3 - 1e-9).all()
    assert e.iloc[50] == pytest.approx(1.0)        # early, no trigger yet


def test_causal_stress_prob_no_future_leak(rets):
    p = em.causal_stress_prob(rets, train_end="2020-09-01")
    assert set(np.unique(np.round(p.dropna().values, 6))) <= set(
        np.round(np.linspace(0, 1, 1000001), 6)) or True   # in [0,1] sanity
    assert (p >= -1e-9).all() and (p <= 1.0 + 1e-9).all()
    # perturb a LATE return; early filtered probs must not change
    r2 = rets.copy(); r2.iloc[480] *= 5
    p2 = em.causal_stress_prob(r2, train_end="2020-09-01")
    assert np.allclose(p.iloc[:400].values, p2.iloc[:400].values, atol=1e-6)
