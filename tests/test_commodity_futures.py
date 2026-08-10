"""
Offline tests for the pure futures logic in commodity_futures.py — the roll /
back-adjustment / carry math (the ibapi fetches are integration, run vs live TWS).

These are the calculations that, if wrong, silently corrupt a futures backtest:
the carry slope, front/next contract selection, and — most importantly — that
back-adjustment removes the artificial ROLL GAP while preserving real returns.
"""
import numpy as np
import pandas as pd
import pytest

import commodity_futures as cf


# --- annualized_carry ------------------------------------------------------
def test_carry_backwardation_positive_contango_negative():
    # front > next (backwardation) -> positive; front < next (contango) -> negative
    assert cf.annualized_carry(80.0, 78.0, 30) > 0
    assert cf.annualized_carry(78.0, 80.0, 30) < 0


def test_carry_formula_value():
    # (80-78)/78 * 365/30
    got = cf.annualized_carry(80.0, 78.0, 30)
    assert got == pytest.approx((80 - 78) / 78 * (365 / 30))


def test_carry_guards_bad_inputs():
    assert np.isnan(cf.annualized_carry(80, 0, 30))       # zero next price
    assert np.isnan(cf.annualized_carry(80, 78, 0))       # zero span


# --- pick_front_next -------------------------------------------------------
def test_pick_front_next_skips_near_expiry():
    exps = ["20260820", "20260922", "20261020"]
    front, nxt = cf.pick_front_next(exps, pd.Timestamp("2026-08-17"), min_days=5)
    # Aug-20 is only 3 days out (< min_days) -> front should be Sep, next Oct
    assert front == "20260922" and nxt == "20261020"


def test_pick_front_next_normal():
    exps = ["20260820", "20260922", "20261020"]
    front, nxt = cf.pick_front_next(exps, pd.Timestamp("2026-08-01"), min_days=5)
    assert front == "20260820" and nxt == "20260922"


def test_pick_front_next_too_few_returns_none():
    front, nxt = cf.pick_front_next(["20260820"], pd.Timestamp("2026-08-01"))
    assert front is None and nxt is None


# --- back_adjust (the critical one) ----------------------------------------
def _mk(dates, closes):
    return pd.DataFrame({"close": closes},
                        index=pd.to_datetime(dates))


def test_back_adjust_removes_roll_gap_preserves_returns():
    d1, d2, d3, d4 = "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"
    # E1 held through d3, then roll into E2 (which is CHEAPER -> a downward gap)
    E1 = _mk([d1, d2, d3], [100.0, 101.0, 102.0])
    E2 = _mk([d3, d4],     [99.0, 100.0])          # E2 also trades on d3 (overlap)
    cont = cf.back_adjust({"20260107": E1, "20260108": E2},
                          roll_dates={"20260107": pd.Timestamp(d3)})
    # continuous series has NO artificial jump at the roll:
    # the d3->d4 return must equal E2's real move (99->100 = +1.01%), not the
    # raw gap (102 -> 100).
    ret = cont.sort_index().pct_change()
    assert ret.loc[pd.Timestamp(d4)] == pytest.approx((100 - 99) / 99, rel=1e-6)
    # earlier within-contract returns preserved (d2->d3 was +1/101 in E1)
    assert ret.loc[pd.Timestamp(d3)] == pytest.approx(1 / 101, rel=1e-6)
    # and the series is continuous (strictly increasing here, no -3 gap)
    assert cont.is_monotonic_increasing


def test_back_adjust_single_contract_is_identity():
    E1 = _mk(["2026-01-05", "2026-01-06"], [50.0, 51.0])
    cont = cf.back_adjust({"20260106": E1}, roll_dates={})
    assert list(cont.values) == [50.0, 51.0]


# --- build_carry_series ----------------------------------------------------
def test_build_carry_series_matches_formula():
    idx = pd.to_datetime(["2026-01-05", "2026-01-06"])
    ff = pd.DataFrame({"front_px": [80.0, 81.0], "next_px": [78.0, 80.0],
                       "days_between": [30, 30]}, index=idx)
    cs = cf.build_carry_series(ff)
    assert cs.iloc[0] == pytest.approx(cf.annualized_carry(80, 78, 30))
    assert cs.iloc[1] == pytest.approx(cf.annualized_carry(81, 80, 30))


# --- universe integrity ----------------------------------------------------
def test_universe_has_sectors_and_symbols():
    assert len(cf.FUTURES) >= 12
    for name, c in cf.FUTURES.items():
        assert c["symbol"] and c["exchange"] and c["ccy"] == "USD"
        assert c["sector"] in {"energy", "metals", "grains", "softs", "meats"}


# --- databento clean front returns (roll-gap handling) ---------------------
def test_front_returns_skips_roll_gap():
    import databento_curve as dbc
    import numpy as np, pandas as pd
    # one market, two contracts: within-contract move real, roll-day gap spurious
    dates = pd.to_datetime(["2020-01-02","2020-01-03","2020-01-06","2020-01-07"])
    raw = pd.DataFrame({
        "ts_event": dates,
        "symbol": ["CL.n.0"]*4,
        "instrument_id": [111,111,222,222],   # roll happens on 3rd row
        "close": [100.0, 101.0, 80.0, 81.0],  # 101->80 is the roll GAP (not a return)
    }).set_index("ts_event")
    m = dbc.front_returns(raw)
    daily_prod = (1+m.fillna(0)).prod()-1
    # real returns: +1% (day2) and +1.25% (day4); the -20% roll gap must be excluded
    assert daily_prod["Oil"] == pytest.approx((1.01*1.0125)-1, rel=1e-6)
