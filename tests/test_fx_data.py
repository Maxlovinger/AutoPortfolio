"""
Offline tests for fx/data.py. Network calls (yfinance, FRED) are monkeypatched
so nothing here touches the internet; we test the transforms, orientation, and
alignment logic — the parts that can silently corrupt a backtest.
"""
import numpy as np
import pandas as pd
import pytest

import fx.data as fxd


# --- carry_base ------------------------------------------------------------
def test_carry_base_is_differential_vs_usd():
    idx = pd.date_range("2020-01-31", periods=3, freq="M")
    rates = pd.DataFrame(
        {"USD": [2.0, 2.0, 2.0], "EUR": [0.0, 0.5, 1.0], "AUD": [4.0, 4.0, 4.0]},
        index=idx,
    )
    carry = fxd.carry_base(rates, base="USD")
    assert "USD" not in carry.columns          # base dropped
    assert list(carry["EUR"]) == [-2.0, -1.5, -1.0]   # below USD = negative
    assert list(carry["AUD"]) == [2.0, 2.0, 2.0]      # above USD = positive


def test_carry_base_base_column_removed_even_if_named_differently():
    idx = pd.date_range("2020-01-31", periods=2, freq="M")
    rates = pd.DataFrame({"GBP": [3.0, 3.0], "JPY": [0.0, 0.0]}, index=idx)
    carry = fxd.carry_base(rates, base="JPY")
    assert list(carry.columns) == ["GBP"]
    assert list(carry["GBP"]) == [3.0, 3.0]


# --- fetch_short_rates (monkeypatched FRED) --------------------------------
class _FakeFred:
    def __init__(self, *a, **k):
        pass

    def get_series(self, sid, observation_start=None):
        # daily series; different constant per id so we can tell them apart
        idx = pd.date_range("2020-01-01", periods=90, freq="D")
        val = 1.0 if "US" in sid else 3.0
        return pd.Series(val, index=idx)


def test_fetch_short_rates_resamples_to_month_end(monkeypatch):
    monkeypatch.setattr(fxd, "Fred", _FakeFred)
    monkeypatch.setattr(fxd, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("FRED_API", "dummy")
    uni = {"USD": {"fred": "IR3TIB01USM156N"},
           "EUR": {"fred": "IR3TIB01EZM156N"}}
    rates = fxd.fetch_short_rates(universe=uni)
    assert list(rates.columns) == ["USD", "EUR"]
    # monthly resample of ~90 daily points -> 3-4 month-end rows
    assert 3 <= len(rates) <= 4
    assert (rates["USD"] == 1.0).all() and (rates["EUR"] == 3.0).all()


def test_fetch_short_rates_missing_key_raises(monkeypatch):
    monkeypatch.setattr(fxd, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("FRED_API", raising=False)
    with pytest.raises(RuntimeError, match="FRED_API"):
        fxd.fetch_short_rates()


# --- fetch_spot orientation (monkeypatched yfinance) -----------------------
def _fake_yf_download(tickers, **kwargs):
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    # EURUSD quoted USD-per-EUR already; USDJPY quoted JPY-per-USD (needs invert)
    close = pd.DataFrame(
        {"EURUSD=X": np.linspace(1.10, 1.14, 5),
         "USDJPY=X": np.linspace(100.0, 125.0, 5)},
        index=idx,
    )
    return pd.concat({"Close": close}, axis=1)


def test_fetch_spot_orientation_and_inversion(monkeypatch):
    monkeypatch.setattr(fxd.yf, "download", _fake_yf_download)
    uni = {
        "USD": {"ticker": None, "invert": False},
        "EUR": {"ticker": "EURUSD=X", "invert": False},
        "JPY": {"ticker": "USDJPY=X", "invert": True},
    }
    spot = fxd.fetch_spot(universe=uni)
    assert "USD" not in spot.columns                 # base has no spot column
    # EUR passed through as USD-per-EUR
    assert spot["EUR"].iloc[0] == pytest.approx(1.10)
    # JPY inverted: 1/100 -> 1/125, i.e. USD-per-JPY, and it FALLS as yen weakens
    assert spot["JPY"].iloc[0] == pytest.approx(1 / 100.0)
    assert spot["JPY"].iloc[-1] == pytest.approx(1 / 125.0)
    assert spot["JPY"].iloc[-1] < spot["JPY"].iloc[0]


def test_monthly_fx_returns_shape():
    idx = pd.bdate_range("2020-01-01", periods=70)
    spot = pd.DataFrame({"EUR": np.linspace(1.0, 1.1, 70),
                         "JPY": np.linspace(0.01, 0.009, 70)}, index=idx)
    r = fxd.monthly_fx_returns(spot)
    assert list(r.columns) == ["EUR", "JPY"]
    assert r["EUR"].iloc[-1] > 0          # rising spot -> positive return
    assert r["JPY"].iloc[-1] < 0          # falling spot -> negative return
