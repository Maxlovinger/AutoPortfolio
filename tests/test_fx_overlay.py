"""
Offline tests for fx/overlay.py — the vol-target crash overlay. Synthetic
return series so causality (no look-ahead), the de-risking response to vol
spikes, cost accounting, and the cap all check exactly.
"""
import numpy as np
import pandas as pd
import pytest

import fx.overlay as ov
from fx.backtest_carry import run_carry_backtest, summarize
from tests.test_fx_carry import make_carry_world


def _monthly(vals):
    idx = pd.date_range("2015-01-31", periods=len(vals), freq="M")
    return pd.Series(vals, index=idx)


# --- realized_vol ----------------------------------------------------------
def test_realized_vol_rises_with_dispersion():
    calm = _monthly([0.001] * 24)
    rv = ov.realized_vol(calm, lookback=6, ppy=12).dropna()
    assert (rv < 1e-6).all()                         # constant -> ~0 vol
    noisy = _monthly(list(np.random.default_rng(0).normal(0, 0.05, 24)))
    rv2 = ov.realized_vol(noisy, lookback=6, ppy=12).dropna()
    assert rv2.mean() > 0.05


# --- exposure: causal + de-risks on vol spikes -----------------------------
def test_exposure_cuts_after_vol_spike():
    # calm then a burst of high vol; exposure should FALL after the burst
    r = _monthly([0.005] * 12 + list(np.random.default_rng(1).normal(0, 0.08, 12)))
    k = ov.vol_target_exposure(r, target_vol=0.05, freq="M",
                               lookback_months=6, cap=5.0)
    assert k.iloc[8] > k.iloc[-1]                    # calm exposure > stressed


def test_exposure_is_causal_no_lookahead():
    # a single huge shock at t: exposure AT t must not yet reflect it (shift 1)
    r = _monthly([0.005] * 10 + [0.30] + [0.005] * 6)
    k = ov.vol_target_exposure(r, target_vol=0.05, freq="M",
                               lookback_months=6, cap=5.0)
    shock = r.index[10]
    post = r.index[11]
    # exposure only drops the period AFTER the shock enters the vol window
    assert k.loc[post] < k.loc[shock]


def test_exposure_respects_cap():
    r = _monthly([0.0001] * 24)                      # near-zero vol -> huge ratio
    k = ov.vol_target_exposure(r, target_vol=0.05, freq="M",
                               lookback_months=6, cap=2.5)
    assert k.max() <= 2.5 + 1e-12


def test_exposure_never_negative():
    r = _monthly(list(np.random.default_rng(2).normal(0, 0.05, 30)))
    k = ov.vol_target_exposure(r, target_vol=0.05, freq="M")
    assert (k >= 0).all()


# --- apply_overlay costs ---------------------------------------------------
def test_apply_overlay_charges_for_resizing():
    r = _monthly([0.01] * 12)
    exp = _monthly([1.0, 2.0] + [1.0] * 10)          # one big resize at t=1
    net, k = ov.apply_overlay(r, exp, cost_bps=100.0)
    # at the resize period, net < scaled gross by the resize cost
    assert net.iloc[1] < 2.0 * 0.01
    # steady exposure -> no resize cost
    assert net.iloc[5] == pytest.approx(1.0 * 0.01, abs=1e-12)


def test_apply_overlay_scales_returns():
    r = _monthly([0.02] * 6)
    exp = _monthly([0.5] * 6)
    net, k = ov.apply_overlay(r, exp, cost_bps=0.0)
    assert np.allclose(net.values, 0.01)             # half exposure -> half ret


# --- integration on a backtest result --------------------------------------
def test_run_vol_target_shape_and_summarize():
    spot, carry = make_carry_world()
    base = run_carry_backtest(spot, carry, freq="M", n_long=2, n_short=2)
    res = ov.run_vol_target(base, target_vol=0.05, lookback_months=6)
    assert set(["net_ret", "equity", "exposure", "freq"]).issubset(res)
    m = summarize(res)
    assert m["n"] > 0
    assert res["exposure"].notna().all()


def test_overlay_reduces_drawdown_on_crash_shaped_book():
    # build a return series: steady gains then a violent crash (carry-crash shape)
    idx = pd.date_range("2015-01-31", periods=40, freq="M")
    r = pd.Series([0.01] * 30 + [-0.02, -0.03, -0.08, -0.02] + [0.01] * 6, index=idx)
    base = {"freq": "M", "net_ret": r, "gross_ret": r,
            "turnover": pd.Series(0.0, index=idx),
            "equity": (1 + r).cumprod()}
    over = ov.run_vol_target(base, target_vol=0.05, lookback_months=6, cap=2.0)
    base_dd = summarize(base)["max_dd"]
    over_dd = summarize(over)["max_dd"]
    # vol-targeting should de-risk into the crash -> shallower drawdown
    assert over_dd > base_dd                          # less negative
