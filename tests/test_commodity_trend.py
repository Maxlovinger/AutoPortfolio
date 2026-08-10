"""
Offline tests for the trend backtest (commodity_trend.py). The ibapi/data load is
integration; here the signal, sizing, no-lookahead, and vol-target mechanics are
checked on synthetic monthly returns with known structure.
"""
import numpy as np
import pandas as pd
import pytest

import commodity_trend as ct
from utils import MONTH_END


def _panel(n=90, seed=0):
    idx = pd.date_range("2015-01-31", periods=n, freq=MONTH_END)
    rng = np.random.default_rng(seed)
    # A trends UP, B trends DOWN, C is calm-low-vol, D is high-vol noise
    A = rng.normal(0.015, 0.04, n)
    B = rng.normal(-0.012, 0.04, n)
    C = rng.normal(0.0, 0.01, n)
    D = rng.normal(0.0, 0.10, n)
    return pd.DataFrame({"A": A, "B": B, "C": C, "D": D}, index=idx)


# --- signal ----------------------------------------------------------------
def test_trend_score_lagged_and_bounded():
    r = _panel(seed=1)
    s = ct.trend_score(r, lookbacks=(3, 6, 12))
    assert s.abs().max().max() <= 1.0 + 1e-9          # average of signs in [-1,1]
    # first 12 rows can't be scored (needs 12m + the shift)
    assert s.iloc[:12].isna().all().all()


def test_trend_score_no_lookahead():
    r = _panel(seed=2)
    s1 = ct.trend_score(r)
    r2 = r.copy(); r2.iloc[40] += 5.0                 # shock month 40's return
    s2 = ct.trend_score(r2)
    # month 40's OWN score uses data through 39 -> unchanged by the shock at 40
    assert s1.iloc[40].equals(s2.iloc[40])


# --- inverse-vol sizing ----------------------------------------------------
def test_inv_vol_smaller_weight_for_higher_vol():
    r = _panel(seed=3)
    iv = ct.inv_vol(r, lookback=12).dropna()
    # D (10% monthly vol) must get a much smaller inverse-vol number than C (1%)
    assert iv["C"].iloc[-1] > iv["D"].iloc[-1]


def test_weights_risk_normalized_to_gross_one():
    r = _panel(seed=4)
    w = ct.weights(r).dropna(how="all")
    gross = w.abs().sum(axis=1).dropna()
    assert np.allclose(gross[gross > 0], 1.0, atol=1e-9)


# --- backtest --------------------------------------------------------------
def test_backtest_no_lookahead_prior_returns_stable():
    r = _panel(seed=5)
    base = ct.backtest(r, vol_target=False)["net"]
    r2 = r.copy(); r2.iloc[75] += 3.0                 # shock a LATE month
    shocked = ct.backtest(r2, vol_target=False)["net"]
    # compare only months whose date is strictly before the shocked month;
    # a return at month 75 must not alter any earlier net return
    shock_date = r.index[75]
    common = base.index[base.index < shock_date]
    pd.testing.assert_series_equal(base.loc[common], shocked.loc[common])


def test_trend_book_profits_on_persistent_trends():
    # A up, B down -> a long/short trend book should be net positive
    r = _panel(seed=6)
    net = ct.backtest(r, vol_target=False, cost_bps=0.0)["net"]
    assert net.mean() > 0


def test_vol_target_scales_toward_target():
    r = _panel(seed=7)
    res = ct.backtest(r, vol_target=True, target_vol=0.12, cap=5.0)
    v = ct.performance(res["net"])["vol"]
    # realized vol should land in a reasonable band around the 12% target
    assert 0.04 < v < 0.30


def test_costs_reduce_return():
    r = _panel(seed=8)
    free = ct.backtest(r, vol_target=False, cost_bps=0.0)["net"].sum()
    charged = ct.backtest(r, vol_target=False, cost_bps=50.0)["net"].sum()
    assert charged < free


# --- metrics ---------------------------------------------------------------
def test_performance_reports_skew_and_dd():
    r = pd.Series([0.01] * 30 + [-0.25],
                  index=pd.date_range("2015-01-31", periods=31, freq=MONTH_END))
    m = ct.performance(r)
    assert m["skew"] < 0 and m["maxdd"] < 0 and m["worst"] == pytest.approx(-0.25)
