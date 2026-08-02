"""
Tests for volatility forecasts + the exposure overlay. Offline/deterministic.
Emphasis on the no-look-ahead property (a forecast for day t must not depend on
day t's own return).
"""
import numpy as np
import pandas as pd
import pytest

from vol_forecast import (realized_vol, ewma_vol, vix_forecast, blend,
                          exposure, apply_overlay, make_forecast)


@pytest.fixture
def rets():
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.005, 200)
    wild = rng.normal(0, 0.04, 200)
    return pd.Series(np.concatenate([calm, wild]),
                     index=pd.date_range("2020-01-01", periods=400, freq="B"))


def test_ewma_rises_after_vol_spike(rets):
    f = ewma_vol(rets)
    assert f.iloc[250] > f.iloc[150]          # wild regime > calm regime


def test_ewma_no_lookahead(rets):
    """Changing r[t] must not change the forecast at or before t."""
    f1 = ewma_vol(rets)
    r2 = rets.copy()
    k = 300
    r2.iloc[k] *= 10                          # perturb one day
    f2 = ewma_vol(r2)
    assert np.allclose(f1.iloc[:k + 1].values, f2.iloc[:k + 1].values)
    assert f2.iloc[k + 1] != f1.iloc[k + 1]   # but future forecast changes


def test_realized_vol_shifted(rets):
    f = realized_vol(rets, lookback=21)
    assert np.isnan(f.iloc[0])                # first value unknown (shifted)


def test_vix_forecast_lagged_and_scaled(rets):
    vix = pd.Series(0.20, index=rets.index)   # constant 20%
    f = vix_forecast(rets, vix, vix_scale=0.8)
    assert np.isnan(f.iloc[0])                # lagged by 1
    assert f.dropna().iloc[0] == pytest.approx(0.16)   # 0.20 * 0.8


def test_blend_between_components(rets):
    vix = pd.Series(0.30, index=rets.index)
    ew = ewma_vol(rets)
    b = blend(rets, vix, w=0.5)
    vx = vix_forecast(rets, vix)
    # where both defined, blend lies between the two inputs
    idx = ew.index[50]
    lo, hi = sorted([ew.loc[idx], vx.loc[idx]])
    assert lo - 1e-9 <= b.loc[idx] <= hi + 1e-9


def test_exposure_capped_and_derisks(rets):
    f = pd.Series([0.10, 0.15, 0.30, 0.60], index=rets.index[:4])
    e = exposure(f, target=0.15, max_lev=1.0)
    assert e.iloc[0] == pytest.approx(1.0)    # low vol -> capped at 1.0
    assert e.iloc[2] == pytest.approx(0.5)    # 0.15/0.30
    assert e.iloc[3] == pytest.approx(0.25)   # 0.15/0.60


def test_apply_overlay_reduces_turbulent_returns(rets):
    f = ewma_vol(rets)
    ov, e = apply_overlay(rets, f, target=0.15)
    # overlaid wild-period vol should be lower than raw wild-period vol
    assert ov.iloc[200:].std() < rets.iloc[200:].std()
    assert (e <= 1.0 + 1e-9).all()


def test_make_forecast_dispatch(rets):
    vix = pd.Series(0.20, index=rets.index)
    for m in ["realized", "ewma", "vix", "blend"]:
        f = make_forecast(rets, method=m, vix=vix)
        assert isinstance(f, pd.Series) and len(f) == len(rets)
    with pytest.raises(ValueError):
        make_forecast(rets, method="nope", vix=vix)
