"""
Offline tests for the commodity signal-evaluation harness (commodity_signals.py).

The harness is the thing that decides whether a news/fundamental factor is real,
so its guarantees are tested hard: strict no-lookahead alignment, correct IC sign,
a tilt that pays when the signal is genuine, and — most importantly — a NULL that
sits at ~0 for random signals but flags a perfect one. That null is exactly what
caught equity/FX sentiment being noise.
"""
import numpy as np
import pandas as pd
import pytest

import commodity_signals as cs
from utils import MONTH_END


def _grid(n=120, cols=("A", "B", "C", "D", "E", "F"), seed=0):
    idx = pd.date_range("2010-01-31", periods=n, freq=MONTH_END)
    rng = np.random.default_rng(seed)
    rets = pd.DataFrame(rng.normal(0, 0.05, (n, len(cols))), index=idx,
                        columns=list(cols))
    return idx, rets


# --- alignment / no lookahead ----------------------------------------------
def test_align_pairs_signal_with_next_month_return():
    idx, rets = _grid()
    sig = rets.copy()                                  # signal == current return
    s, fwd = cs._align(sig, rets)
    # fwd at t must equal rets at t+1 (strictly forward)
    assert fwd.iloc[0].equals(rets.iloc[1]) or np.allclose(
        fwd.iloc[0].values, rets.iloc[1].values)
    # last row's forward return is unknown -> NaN
    assert fwd.iloc[-1].isna().all()


def test_changing_current_return_does_not_leak_into_signal_eval():
    # IC pairs signal_t with return_{t+1}; altering return_t must not change the
    # IC contribution attributed to month t's signal (no contemporaneous leak)
    idx, rets = _grid(seed=1)
    sig = pd.DataFrame(np.random.default_rng(2).normal(0, 1, rets.shape),
                       index=rets.index, columns=rets.columns)
    ic1 = cs.information_coefficient(sig, rets)
    rets2 = rets.copy()
    rets2.iloc[5] = rets2.iloc[5] + 10.0               # shock the contemporaneous month
    ic2 = cs.information_coefficient(sig, rets2)
    # month index 4's IC uses return row 5 (its forward), so it MAY change; but
    # month 5's own IC uses row 6 and must be unchanged
    assert ic1.iloc[5] == pytest.approx(ic2.iloc[5])


# --- IC correctness --------------------------------------------------------
def test_perfect_signal_gives_ic_one():
    idx, rets = _grid(seed=3)
    perfect = rets.shift(-1)                            # signal = next month return
    ic = cs.information_coefficient(perfect, rets)
    assert ic.dropna().mean() > 0.99                    # essentially +1 every month


def test_inverted_signal_gives_negative_ic():
    idx, rets = _grid(seed=4)
    inverted = -rets.shift(-1)
    ic = cs.information_coefficient(inverted, rets)
    assert ic.dropna().mean() < -0.99


def test_random_signal_ic_near_zero():
    idx, rets = _grid(seed=5)
    rng = np.random.default_rng(9)
    noise = pd.DataFrame(rng.normal(0, 1, rets.shape), index=rets.index,
                         columns=rets.columns)
    assert abs(cs.information_coefficient(noise, rets).dropna().mean()) < 0.2


# --- tilt backtest ---------------------------------------------------------
def test_tilt_pays_on_perfect_signal_and_is_flat_costed():
    idx, rets = _grid(seed=6)
    perfect = rets.shift(-1)
    r = cs.tilt_backtest(perfect, rets, mode="ls", frac=0.34)
    assert r.mean() > 0                                 # long winners/short losers pays
    # costs only reduce it
    r_costed = cs.tilt_backtest(perfect, rets, mode="ls", frac=0.34, cost_bps=50)
    assert r_costed.sum() < r.sum()


def test_tilt_long_only_mode_runs():
    idx, rets = _grid(seed=7)
    r = cs.tilt_backtest(rets.shift(-1), rets, mode="long")
    assert len(r) > 0


# --- the NULL --------------------------------------------------------------
def test_permutation_preserves_values_and_nan_structure():
    idx, rets = _grid(n=10)
    sig = rets.copy()
    sig.iloc[3, 2] = np.nan
    rng = np.random.default_rng(0)
    perm = cs._permute_within_month(sig, rng)
    for t in sig.index:
        a = sig.loc[t].dropna().sort_values().values
        b = perm.loc[t].dropna().sort_values().values
        assert np.allclose(a, b)                        # same multiset per row
        assert sig.loc[t].isna().equals(perm.loc[t].isna())  # same NaN pattern


def test_null_flags_perfect_signal_high_z():
    idx, rets = _grid(seed=8)
    res = cs.null_test(rets.shift(-1), rets, n_null=100, seed=0)
    assert res["ic_z"] > 3                              # far outside the null
    assert res["ic_pctile"] > 0.98


def test_null_random_signal_z_near_zero():
    idx, rets = _grid(seed=10)
    rng = np.random.default_rng(11)
    noise = pd.DataFrame(rng.normal(0, 1, rets.shape), index=rets.index,
                         columns=rets.columns)
    res = cs.null_test(noise, rets, n_null=150, seed=1)
    assert abs(res["ic_z"]) < 2.5                       # indistinguishable from null


def test_evaluate_signal_verdict_true_for_real_false_for_noise():
    idx, rets = _grid(seed=12)
    good = cs.evaluate_signal(rets.shift(-1), rets, name="perfect", n_null=100)
    assert good["passes"] is True
    rng = np.random.default_rng(3)
    noise = pd.DataFrame(rng.normal(0, 1, rets.shape), index=rets.index,
                         columns=rets.columns)
    bad = cs.evaluate_signal(noise, rets, name="noise", n_null=100)
    assert bad["passes"] is False


# --- time-series / pooled path (narrow, sector-specific signals) -----------
def test_pooled_ic_perfect_narrow_signal():
    # a single-commodity signal that equals its own next-month return -> pooled IC ~1
    idx, rets = _grid(seed=20)
    narrow = rets[["A"]].shift(-1)                     # 1 column only
    assert cs.pooled_ic(narrow, rets[["A"]]) > 0.99


def test_pooled_ic_random_near_zero():
    idx, rets = _grid(seed=21)
    rng = np.random.default_rng(1)
    narrow = pd.DataFrame({"A": rng.normal(0, 1, len(idx))}, index=idx)
    assert abs(cs.pooled_ic(narrow, rets[["A"]])) < 0.25


def test_timeseries_tilt_pays_on_perfect_signal():
    idx, rets = _grid(seed=22)
    narrow = rets[["A", "B"]].shift(-1)
    r = cs.timeseries_tilt(narrow, rets[["A", "B"]])
    assert r.mean() > 0                                # sign-follows-future pays


def test_permute_within_commodity_preserves_column_multiset():
    idx, rets = _grid(n=15, seed=23)
    sig = rets.copy(); sig.iloc[4, 1] = np.nan
    perm = cs._permute_within_commodity(sig, np.random.default_rng(0))
    for c in sig.columns:
        a = sig[c].dropna().sort_values().values
        b = perm[c].dropna().sort_values().values
        assert np.allclose(a, b)
        assert sig[c].isna().equals(perm[c].isna())


def test_null_test_ts_flags_perfect_and_clears_random():
    idx, rets = _grid(seed=24)
    good = cs.null_test_ts(rets[["A"]].shift(-1), rets[["A"]], n_null=100)
    assert good["pooled_ic_z"] > 3
    rng = np.random.default_rng(2)
    noise = pd.DataFrame({"A": rng.normal(0, 1, len(idx))}, index=idx)
    bad = cs.null_test_ts(noise, rets[["A"]], n_null=100)
    assert abs(bad["pooled_ic_z"]) < 2.5


def test_evaluate_signal_narrow_uses_timeseries_and_passes():
    # a 1-column perfect signal has NO cross-section but must still evaluate + pass
    idx, rets = _grid(seed=25)
    out = cs.evaluate_signal(rets[["A"]].shift(-1), rets[["A"]],
                             name="narrow", n_null=100)
    assert "xs_ic_z" not in out                        # no cross-section attempted
    assert out["pooled_ic_z"] > 3 and out["passes"] is True


def test_evaluate_all_ranks_by_ic_z():
    idx, rets = _grid(seed=13)
    rng = np.random.default_rng(4)
    noise = pd.DataFrame(rng.normal(0, 1, rets.shape), index=rets.index,
                         columns=rets.columns)
    out = cs.evaluate_all({"perfect": rets.shift(-1), "noise": noise},
                          rets, n_null=80)
    assert list(out.index)[0] == "perfect"             # best signal ranked first
