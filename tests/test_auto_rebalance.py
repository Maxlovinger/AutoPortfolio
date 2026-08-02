"""
Tests for the automated live job's pure decision logic (no I/O, no network).
"""
import numpy as np
import pandas as pd
import pytest

import auto_rebalance as ar


def test_first_run_is_rebalance_day():
    assert ar.is_rebalance_day("2026-08-03", None) is True


def test_rebalance_only_after_a_quarter():
    assert ar.is_rebalance_day("2026-09-01", "2026-08-03") is False   # ~4 weeks
    assert ar.is_rebalance_day("2026-11-10", "2026-08-03") is True    # ~99 days


def test_exposure_check_on_mondays():
    assert ar.is_exposure_check_day("2026-08-03") is True    # a Monday
    assert ar.is_exposure_check_day("2026-08-05") is False   # a Wednesday


def test_decide_exposure_calm_stays_full():
    # low vol -> target 100%, already at 100% -> no change
    new, changed, why = ar.decide_exposure(0.10, 1.0)
    assert new == 1.0 and changed is False
    assert "NO CHANGE" in why


def test_decide_exposure_spike_cuts():
    # vol 30% -> target 50%; held 100%; 50pt move > 10% band -> resize
    new, changed, why = ar.decide_exposure(0.30, 1.0)
    assert changed is True
    assert new == pytest.approx(0.5, abs=1e-6)
    assert "RE-SIZED" in why


def test_decide_exposure_small_move_within_band():
    # vol 16% -> target ~0.9375; held 1.0 -> 6pt move < 10% band -> hold
    new, changed, why = ar.decide_exposure(0.16, 1.0)
    assert changed is False and new == 1.0


def test_decide_exposure_missing_vol_holds():
    new, changed, why = ar.decide_exposure(float("nan"), 0.7)
    assert new == 0.7 and changed is False


def test_select_book_respects_sector_cap():
    adv = pd.Series({f"T{i}": 100 - i for i in range(30)})
    secs = ["Tech", "Health", "Financials"]
    sectors = {t: secs[i % 3] for i, t in enumerate(adv.index)}
    valid = set(adv.index)
    picks = ar.select_book(adv, sectors, valid, n=8, cap=3)
    assert len(picks) == 8
    for s in secs:
        assert sum(1 for t in picks if sectors[t] == s) <= 3


def test_select_book_skips_unpriced():
    adv = pd.Series({"A": 10, "B": 9, "C": 8})
    sectors = {"A": "X", "B": "Y", "C": "Z"}
    picks = ar.select_book(adv, sectors, valid={"B", "C"}, n=5, cap=5)
    assert "A" not in picks and set(picks) == {"B", "C"}


def test_target_weights_scaled_by_exposure():
    w = ar.target_weights(["A", "B", "C", "D"], exposure=0.6)
    assert abs(w.sum() - 0.6) < 1e-9        # rest is cash
    assert all(abs(v - 0.15) < 1e-9 for v in w)


def test_target_weights_empty():
    assert ar.target_weights([], 1.0).empty


def test_realized_vol_needs_history():
    assert np.isnan(ar.realized_vol(pd.Series([0.01, 0.02])))
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 60))
    assert ar.realized_vol(r) > 0
