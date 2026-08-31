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


# --- weekly-hold: holdings rebalance quarterly, not every week ---------------
_ORDERS = [
    {"ticker": "AAPL", "shares": 1, "action": "BUY", "price": 200.0},
    {"ticker": "NVDA", "shares": -2, "action": "SELL", "price": 180.0},
    {"ticker": "IEF", "shares": 3, "action": "BUY", "price": 95.0},
]


def test_weekly_hold_passes_through_when_rebalancing():
    # quarterly rebalance OR an exposure change -> equity IS allowed to move
    out = ar.weekly_equity_hold(list(_ORDERS), "IEF", rebalance_equity=True)
    assert out == _ORDERS


def test_weekly_hold_keeps_only_ief_when_flat():
    rationale = []
    out = ar.weekly_equity_hold(list(_ORDERS), "IEF", rebalance_equity=False,
                                rationale=rationale)
    assert [o["ticker"] for o in out] == ["IEF"]        # equity drift suppressed
    assert "suppressed 2 equity drift" in rationale[0]


def test_weekly_hold_empty_when_no_ief_trade():
    eq_only = [o for o in _ORDERS if o["ticker"] != "IEF"]
    out = ar.weekly_equity_hold(eq_only, "IEF", rebalance_equity=False)
    assert out == []


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


# ---------------------------------------------------------------- dual-class dedup
def test_base_company_strips_share_class():
    assert ar._base_company("Alphabet Inc. (Class C)") == "Alphabet Inc."
    assert ar._base_company("Alphabet Inc. (Class A)") == "Alphabet Inc."
    assert ar._base_company("Apple Inc.") == "Apple Inc."          # unchanged


def test_select_book_dedup_keeps_most_liquid_class():
    # GOOG more liquid than GOOGL; both are Alphabet -> keep only GOOG (higher ADV)
    adv = pd.Series({"GOOG": 100, "GOOGL": 90, "AAPL": 80, "MSFT": 70})
    sectors = {t: "Comm" if t.startswith("GOOG") else "Tech" for t in adv.index}
    names = {"GOOG": "Alphabet Inc. (Class C)", "GOOGL": "Alphabet Inc. (Class A)",
             "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp."}
    picks = ar.select_book(adv, sectors, valid=set(adv.index), n=3, cap=5, names=names)
    assert "GOOG" in picks and "GOOGL" not in picks               # one class only
    assert picks == ["GOOG", "AAPL", "MSFT"]


def test_select_book_without_names_keeps_both_classes():
    # backward compatible: no names dict -> no dedup (old behaviour)
    adv = pd.Series({"GOOG": 100, "GOOGL": 90, "AAPL": 80})
    sectors = {t: "X" for t in adv.index}
    picks = ar.select_book(adv, sectors, valid=set(adv.index), n=3, cap=5)
    assert "GOOG" in picks and "GOOGL" in picks


# ---------------------------------------------------------------- position reconcile
def test_normalize_positions_space_to_dash_and_int():
    raw = {"BRK B": 2.0, "AAPL": 4.0, "BF B": 3.0}
    out = ar.normalize_positions(raw)
    assert out == {"BRK-B": 2, "AAPL": 4, "BF-B": 3}
    assert all(isinstance(v, int) for v in out.values())


def test_normalize_positions_fractional_keeps_float():
    out = ar.normalize_positions({"BRK B": 2.5}, fractional=True)
    assert out == {"BRK-B": 2.5}
    assert isinstance(out["BRK-B"], float)


def test_brk_b_not_rebought_after_reconcile():
    # REGRESSION: held 'BRK B' must reconcile against target 'BRK-B' so a book
    # already holding it generates NO order (previously it re-bought, overweighting).
    from paper_trader import plan_orders
    weights = ar.target_weights(["BRK-B", "AAPL"], exposure=1.0)   # 0.5 each
    prices = pd.Series({"BRK-B": 500.0, "AAPL": 250.0})
    nav = 2000.0                                    # target: BRK-B 2 sh, AAPL 4 sh
    current = ar.normalize_positions({"BRK B": 2.0, "AAPL": 4.0})
    trades = plan_orders(weights, prices, nav, current)
    assert trades == []                             # already on target, no re-buy
