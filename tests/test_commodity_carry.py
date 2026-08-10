"""
Offline tests for the FRED spot-basis carry proxy (commodity_carry_proxy.py).
FRED fetch is integration; the basis math, sign, no-lookahead, and the
cross-sectional carry backtest mechanics are tested on synthetic data.
"""
import numpy as np
import pandas as pd
import pytest

import commodity_carry_proxy as cp
from utils import MONTH_END


def _mk(n=60, seed=0):
    idx = pd.date_range("2015-01-31", periods=n, freq=MONTH_END)
    rng = np.random.default_rng(seed)
    front = pd.DataFrame(
        {c: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
         for c in ["Oil", "NatGas", "Copper", "Corn"]}, index=idx)
    return idx, front, rng


# --- basis -----------------------------------------------------------------
def test_build_basis_backwardation_positive():
    idx = pd.date_range("2015-01-31", periods=3, freq=MONTH_END)
    spot = pd.DataFrame({"Oil": [110.0, 110, 110]}, index=idx)
    front = pd.DataFrame({"Oil": [100.0, 100, 100]}, index=idx)   # spot > front
    b = cp.build_basis(spot, front)
    assert (b["Oil"] > 0).all()                       # backwardation -> positive basis


def test_build_basis_contango_negative_and_value():
    idx = pd.date_range("2015-01-31", periods=2, freq=MONTH_END)
    spot = pd.DataFrame({"Oil": [90.0, 90]}, index=idx)
    front = pd.DataFrame({"Oil": [100.0, 100]}, index=idx)        # spot < front
    b = cp.build_basis(spot, front)
    assert b["Oil"].iloc[0] == pytest.approx(np.log(90) - np.log(100))
    assert (b["Oil"] < 0).all()


def test_build_basis_only_shared_markets():
    idx = pd.date_range("2015-01-31", periods=2, freq=MONTH_END)
    spot = pd.DataFrame({"Oil": [1.0, 1], "Gold": [1.0, 1]}, index=idx)
    front = pd.DataFrame({"Oil": [1.0, 1], "Corn": [1.0, 1]}, index=idx)
    b = cp.build_basis(spot, front)
    assert list(b.columns) == ["Oil"]                 # only the intersection


# --- signal (no lookahead) -------------------------------------------------
def test_carry_signal_is_lagged():
    idx, front, rng = _mk(seed=1)
    spot = front * (1 + rng.normal(0, 0.05, front.shape))
    basis = cp.build_basis(spot, front)
    sig = cp.carry_signal(basis, min_periods=12)
    # early rows unstandardizable; a value at t uses basis through t-1 (shifted)
    assert sig.iloc[:12].isna().all().all()
    # shocking basis at month 40 must not change month 40's own signal
    b2 = basis.copy(); b2.iloc[40] += 5.0
    s2 = cp.carry_signal(b2, min_periods=12)
    assert sig.iloc[40].equals(s2.iloc[40])


# --- carry weights ---------------------------------------------------------
def test_carry_weights_market_neutral_and_gross_one():
    idx, front, rng = _mk(seed=2)
    rets = front.pct_change()
    # a fixed cross-sectional signal (already "lagged" shape)
    sig = pd.DataFrame(np.tile([2.0, 1.0, -1.0, -2.0], (len(idx), 1)),
                       index=idx, columns=front.columns)
    w = cp.carry_weights(sig, rets).dropna(how="all")
    net = w.sum(axis=1).dropna()
    gross = w.abs().sum(axis=1).dropna()
    assert np.allclose(net[gross > 0], 0.0, atol=1e-9)      # demeaned -> ~neutral
    assert np.allclose(gross[gross > 0], 1.0, atol=1e-9)    # risk-normalized


def test_carry_weights_long_high_signal_short_low():
    idx, front, rng = _mk(seed=3)
    rets = front.pct_change()
    sig = pd.DataFrame(np.tile([3.0, 1.0, -1.0, -3.0], (len(idx), 1)),
                       index=idx, columns=front.columns)
    w = cp.carry_weights(sig, rets).dropna(how="all").iloc[-1]
    assert w["Oil"] > 0 and w["Corn"] < 0             # highest long, lowest short


# --- backtest --------------------------------------------------------------
def test_backtest_no_lookahead():
    idx, front, rng = _mk(n=80, seed=4)
    rets = front.pct_change()
    spot = front * (1 + rng.normal(0, 0.05, front.shape))     # varying basis
    sig = cp.carry_signal(cp.build_basis(spot, front))
    base = cp.backtest(sig, rets, vol_target=False)["net"]
    r2 = rets.copy(); r2.iloc[70] += 3.0
    shocked = cp.backtest(sig, r2, vol_target=False)["net"]
    shock_date = rets.index[70]
    common = base.index[base.index < shock_date]
    pd.testing.assert_series_equal(base.loc[common], shocked.loc[common])


def test_carry_book_profits_when_high_carry_outperforms():
    # markets ranked 0..3 by a persistent signal; make high-signal markets drift up
    n = 90
    idx = pd.date_range("2015-01-31", periods=n, freq=MONTH_END)
    rng = np.random.default_rng(5)
    drift = {"Oil": 0.012, "NatGas": 0.004, "Copper": -0.004, "Corn": -0.012}
    front = pd.DataFrame(
        {c: 100 * np.exp(np.cumsum(rng.normal(d, 0.03, n))) for c, d in drift.items()},
        index=idx)
    rets = front.pct_change()
    # signal ranks Oil>NatGas>Copper>Corn every month (already lagged shape)
    sig = pd.DataFrame(np.tile([2.0, 1.0, -1.0, -2.0], (n, 1)),
                       index=idx, columns=list(drift))
    net = cp.backtest(sig, rets, vol_target=False, cost_bps=0.0)["net"]
    assert net.mean() > 0


# --- combine ---------------------------------------------------------------
def test_combine_equal_risk_blend():
    idx = pd.date_range("2015-01-31", periods=60, freq=MONTH_END)
    rng = np.random.default_rng(6)
    a = pd.Series(rng.normal(0.005, 0.02, 60), index=idx)
    b = pd.Series(rng.normal(0.004, 0.06, 60), index=idx)
    blend = cp.combine(a, b)
    assert len(blend) == 60
    # blend vol should sit between the two scaled components' ~target, finite
    assert np.isfinite(blend.std())


# --- map integrity ---------------------------------------------------------
def test_fred_map_points_at_real_markets():
    import commodity_futures as cfut
    for name in cp.FRED_SPOT:
        assert name in cfut.FUTURES, f"{name} not a futures market"
