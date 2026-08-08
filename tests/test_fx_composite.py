"""
Offline tests for fx/composite.py — the carry+value+momentum ranker. Synthetic
spot/carry panels so signal construction, no-look-ahead shifting, z-scoring,
NaN-safe combination, and engine reuse are all exactly checkable.
"""
import numpy as np
import pandas as pd
import pytest

import fx.composite as comp
from tests.test_fx_carry import make_carry_world, make_varying_carry


# --- momentum --------------------------------------------------------------
def test_momentum_positive_for_rising_currency():
    idx = pd.bdate_range("2015-01-01", periods=900)
    spot = pd.DataFrame({"UP": np.linspace(1.0, 2.0, 900),
                         "DN": np.linspace(2.0, 1.0, 900)}, index=idx)
    mom = comp.momentum_signal(spot, freq="M", lookback_months=12).dropna()
    assert (mom["UP"] > 0).all()          # steadily rising -> positive momentum
    assert (mom["DN"] < 0).all()          # steadily falling -> negative


def test_momentum_is_shifted_no_lookahead():
    idx = pd.bdate_range("2015-01-01", periods=900)
    spot = pd.DataFrame({"A": np.linspace(1.0, 2.0, 900)}, index=idx)
    px = comp.resample_spot(spot, "M")
    mom = comp.momentum_signal(spot, freq="M", lookback_months=12)
    raw = px.pct_change(12)
    # signal at t must equal the raw trailing return at t-1 (shifted)
    aligned = mom["A"].dropna()
    assert aligned.iloc[0] == pytest.approx(raw["A"].shift(1).dropna().iloc[0])


# --- value -----------------------------------------------------------------
def test_value_is_negative_long_horizon_return():
    idx = pd.bdate_range("2010-01-01", periods=1800)     # ~7y of bdays
    spot = pd.DataFrame({"CHEAP": np.linspace(2.0, 1.0, 1800),   # fell a lot
                         "RICH": np.linspace(1.0, 2.0, 1800)},   # rose a lot
                        index=idx)
    val = comp.value_signal(spot, freq="M", years=5).dropna()
    assert (val["CHEAP"] > 0).all()       # fell -> cheap -> high value score
    assert (val["RICH"] < 0).all()        # rose -> rich -> low value score


# --- z-score ---------------------------------------------------------------
def test_zscore_rows_centered_and_scaled():
    df = pd.DataFrame({"a": [1.0, 10.0], "b": [2.0, 20.0], "c": [3.0, 30.0]})
    z = comp.zscore_rows(df)
    assert z.iloc[0].mean() == pytest.approx(0.0, abs=1e-12)   # centered
    assert z.iloc[0]["a"] < 0 < z.iloc[0]["c"]                 # ordering kept


def test_zscore_zero_dispersion_is_nan_or_zero():
    df = pd.DataFrame({"a": [5.0], "b": [5.0], "c": [5.0]})
    z = comp.zscore_rows(df)
    assert z.iloc[0].isna().all()         # no dispersion -> undefined, not inf


# --- composite combination -------------------------------------------------
def test_composite_weight_zero_excludes_signal():
    spot, carry = make_carry_world()
    # carry-only composite should match ranking on carry alone
    only_carry = comp.composite_score(
        spot, carry, weights={"carry": 1, "momentum": 0, "value": 0})
    assert only_carry.notna().any().any()
    # a period with valid carry: highest-carry name should have highest score
    row = only_carry.dropna(how="all").iloc[-1]
    assert row.idxmax() == "AAA"          # AAA is the top-carry name by design


def test_composite_handles_missing_history_early():
    # early periods have no 5y value / 12m momentum -> should fall back to carry
    spot, carry = make_carry_world(n_days=400)     # <2y, no value history
    score = comp.composite_score(spot, carry)
    # still produces some tradable (non-all-NaN) rows from carry alone
    assert score.notna().any().any()


# --- backtest engine reuse -------------------------------------------------
def test_composite_backtest_runs_and_matches_carry_when_carry_only():
    spot, carry = make_carry_world()
    from fx.backtest_carry import run_carry_backtest
    a = run_carry_backtest(spot, carry, freq="M", n_long=2, n_short=2,
                           cost_bps=0.0)
    b = comp.run_composite_backtest(
        spot, carry, freq="M", n_long=2, n_short=2, cost_bps=0.0,
        weights={"carry": 1, "momentum": 0, "value": 0})
    # ranking on carry-only via the composite path == the carry backtest,
    # once both have enough names to rank each period
    common = a["net_ret"].index.intersection(b["net_ret"].index)
    diff = (a["net_ret"].reindex(common) - b["net_ret"].reindex(common)).abs()
    assert diff.max() < 1e-9


def test_composite_backtest_both_cadences():
    spot, carry = make_varying_carry()
    for f in ("M", "W"):
        res = comp.run_composite_backtest(spot, carry, freq=f,
                                          n_long=2, n_short=2)
        assert len(res["equity"]) > 0
