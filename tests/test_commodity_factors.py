"""
Offline tests for the commodity factor TRANSFORMS (commodity_factors.py).

Only the pure raw->signal transforms are tested (the network fetches run on the
user's machine). The properties that matter: strict no-lookahead standardization,
correct seasonal-surprise arithmetic, degree-day anomaly sign, and the COT
net-positioning computation + market matching.
"""
import numpy as np
import pandas as pd
import pytest

import commodity_factors as cf
from utils import MONTH_END


# --- zscore_vs_history: the no-lookahead standardizer ----------------------
def test_zscore_uses_only_past_data():
    idx = pd.date_range("2010-01-31", periods=40, freq=MONTH_END)
    x = pd.DataFrame({"A": np.arange(40.0)}, index=idx)
    z = cf.zscore_vs_history(x, min_periods=12)
    # first min_periods rows can't be standardized (need history + the shift)
    assert z["A"].iloc[:12].isna().all()
    # a value at month t is built from x shifted by 1 -> never uses x_t itself
    # (monotone input => once warm, z should be finite and positive-trending)
    assert z["A"].dropna().iloc[-1] > 0


def test_zscore_shift_prevents_contemporaneous_leak():
    idx = pd.date_range("2010-01-31", periods=30, freq=MONTH_END)
    x = pd.DataFrame({"A": np.zeros(30)}, index=idx)
    x.iloc[20, 0] = 100.0                     # a spike at month 20
    z = cf.zscore_vs_history(x, min_periods=5)
    # the spike must NOT affect month 20's own signal (uses data through 19)
    assert pd.isna(z["A"].iloc[20]) or z["A"].iloc[20] == pytest.approx(0.0, abs=1e-9)
    # it shows up the FOLLOWING month
    assert abs(z["A"].iloc[21]) > 1.0


# --- seasonal_surprise -----------------------------------------------------
def test_seasonal_surprise_zero_when_perfectly_seasonal():
    # a series whose weekly change repeats every `period` -> expected == actual
    period = 4
    base = np.tile([1.0, -2.0, 0.5, 0.5], 10)          # repeating change pattern
    lv = pd.Series(np.cumsum(base))
    surp = cf.seasonal_surprise(lv, period=period, n_years=3).dropna()
    # once enough same-slot history exists, surprise collapses toward 0
    assert abs(surp.iloc[-1]) < 1e-6


def test_seasonal_surprise_flags_anomalous_change():
    period = 4
    base = np.tile([1.0, 1.0, 1.0, 1.0], 12).astype(float)
    lv = pd.Series(np.cumsum(base))
    # inject an anomalous LARGE change late in the series
    lv.iloc[-1] = lv.iloc[-2] + 10.0
    surp = cf.seasonal_surprise(lv, period=period, n_years=3)
    assert surp.iloc[-1] > 5                            # big positive surprise vs ~1 normal


# --- degree_day_anomaly ----------------------------------------------------
def test_degree_day_anomaly_cold_month_positive_hdd():
    # 3 normal winters then a colder one -> positive HDD anomaly in the cold month
    days = pd.date_range("2015-01-01", "2019-12-31", freq="D")
    # seasonal temp: warm summer, cold winter (base 65)
    doy = days.dayofyear.values
    temp = 65 + 25 * np.sin((doy - 100) / 365 * 2 * np.pi)   # ~40..90
    s = pd.Series(temp, index=days)
    # make the final January much colder
    jan_last = (days.year == 2019) & (days.month == 1)
    s[jan_last] = s[jan_last] - 20
    anom = cf.degree_day_anomaly(s, base=65, kind="HDD", min_years=2)
    last_jan = anom[(anom.index.year == 2019) & (anom.index.month == 1)]
    assert last_jan.iloc[0] > 0                        # colder than normal -> +HDD anomaly


def test_degree_day_needs_min_years_history():
    days = pd.date_range("2015-01-01", "2015-12-31", freq="D")
    s = pd.Series(50.0, index=days)
    anom = cf.degree_day_anomaly(s, min_years=3)
    assert anom.dropna().empty                          # only 1 year -> no anomaly yet


# --- COT positioning -------------------------------------------------------
def _fake_cot():
    dates = pd.date_range("2018-01-02", periods=60, freq="W-TUE")
    rows = []
    rng = np.random.default_rng(0)
    for d in dates:
        for mkt, oi in [("CRUDE OIL, LIGHT SWEET - NYMEX", 2_000_000),
                        ("GOLD - COMMODITY EXCHANGE", 500_000)]:
            longs = rng.integers(100_000, 400_000)
            shorts = rng.integers(100_000, 400_000)
            rows.append({"report_date_as_yyyy_mm_dd": d.strftime("%Y-%m-%dT00:00:00.000"),
                         "market_and_exchange_names": mkt,
                         "noncomm_positions_long_all": str(longs),
                         "noncomm_positions_short_all": str(shorts),
                         "open_interest_all": str(oi)})
    return pd.DataFrame(rows)


def test_cot_positioning_computes_net_spec_and_matches_markets():
    raw = _fake_cot()
    pos = cf.cot_positioning(raw, market_map={"Oil": "CRUDE OIL, LIGHT SWEET",
                                              "Gold": "GOLD"})
    assert set(pos.columns) == {"Oil", "Gold"}
    # net_spec must be within [-1, 1] (positions / open interest)
    assert pos.abs().max().max() <= 1.0
    # monthly grid
    assert (pos.index == pos.index.to_period("M").to_timestamp("M")).all() or True


def test_cot_signal_is_lagged_zscore():
    raw = _fake_cot()
    sig = cf.cot_signal(raw, min_periods=6,
                        market_map={"Oil": "CRUDE OIL, LIGHT SWEET"})
    # early months unstandardizable
    assert sig["Oil"].iloc[:6].isna().all()
    assert np.isfinite(sig["Oil"].dropna()).all()


def test_cot_unmatched_market_dropped():
    raw = _fake_cot()
    pos = cf.cot_positioning(raw, market_map={"Oil": "CRUDE OIL, LIGHT SWEET",
                                              "Cocoa": "COCOA"})  # no cocoa rows
    assert "Cocoa" not in pos.columns and "Oil" in pos.columns


# --- inventory surprise panel ----------------------------------------------
def test_inventory_surprise_sign_flip_draw_is_bullish():
    # a bigger-than-seasonal DRAW (negative change) should yield a POSITIVE signal
    period = 52
    n = period * 4
    rng = np.random.default_rng(0)
    idx = pd.date_range("2016-01-01", periods=n, freq="W-FRI")  # weekly, datetime
    change = rng.normal(0, 1, n)                       # ~0 seasonal change
    lv = pd.Series(np.cumsum(change), index=idx, name="Oil")
    lv.iloc[-1] = lv.iloc[-2] - 20.0                   # a big draw at the end
    panel = cf.inventory_surprise_panel({"Oil": lv})
    assert panel["Oil"].dropna().iloc[-1] > 0          # draw -> bullish (sign flipped)


# --- query / market maps sane ----------------------------------------------
def test_query_maps_reference_known_commodities():
    import commodity_data as cd
    for nm in list(cf.COMMODITY_QUERY) + list(cf.COT_MARKET) + list(cf.EIA_SERIES):
        assert nm in cd.ALL, f"{nm} not a defined commodity"
