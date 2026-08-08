"""
Offline tests for fx/sentiment.py — the currency news-tone signal. Synthetic
spot + tone panels (no network); we check no-look-ahead alignment, z-scoring,
and that the signal trades on the shared engine like carry/composite.
"""
import numpy as np
import pandas as pd
import pytest

import fx.sentiment as fs
from tests.test_fx_carry import make_carry_world


def make_tone_panel(index_ccys, months=30, seed=0):
    idx = pd.date_range("2019-01-31", periods=months, freq="M")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {c: rng.normal(0, 0.3, months) for c in index_ccys}, index=idx)


# --- alignment / no look-ahead ---------------------------------------------
def test_align_tone_is_shifted():
    tone = make_tone_panel(["AAA", "BBB"])
    grid = pd.date_range("2019-01-31", periods=30, freq="M")
    aligned = fs.align_tone(tone, grid, shift=1)
    # first grid row uses prior-month tone -> NaN (nothing before start)
    assert aligned.iloc[0].isna().all()
    # value at t equals the raw tone at t-1
    assert aligned["AAA"].iloc[5] == pytest.approx(tone["AAA"].iloc[4])


def test_sentiment_signal_zscored_rows():
    spot, carry = make_carry_world()
    tone = make_tone_panel(["AAA", "BBB", "CCC", "DDD", "EEE"], months=36)
    sig = fs.sentiment_signal(spot, tone, freq="M").dropna(how="all")
    # each valid row is cross-sectionally centered
    row = sig.dropna().iloc[-1]
    assert abs(row.mean()) < 1e-9


def test_positive_tone_currency_ranks_high():
    spot, carry = make_carry_world()
    # make AAA consistently the most positive-tone currency
    idx = pd.date_range("2019-01-31", periods=36, freq="M")
    tone = pd.DataFrame(0.0, index=idx,
                        columns=["AAA", "BBB", "CCC", "DDD", "EEE"])
    tone["AAA"] = 0.9
    tone["EEE"] = -0.9
    sig = fs.sentiment_signal(spot, tone, freq="M").dropna(how="all")
    row = sig.dropna().iloc[-1]
    assert row.idxmax() == "AAA"
    assert row.idxmin() == "EEE"


# --- backtest engine reuse -------------------------------------------------
def test_sentiment_backtest_runs():
    spot, carry = make_carry_world()
    tone = make_tone_panel(["AAA", "BBB", "CCC", "DDD", "EEE"], months=36)
    res = fs.run_sentiment_backtest(spot, carry, tone, freq="M",
                                    n_long=2, n_short=2)
    assert len(res["equity"]) > 0
    assert (res["net_ret"] <= res["gross_ret"] + 1e-12).all()


def test_sentiment_backtest_both_cadences():
    spot, carry = make_carry_world()
    tone = make_tone_panel(["AAA", "BBB", "CCC", "DDD", "EEE"], months=36)
    for f in ("M", "W"):
        res = fs.run_sentiment_backtest(spot, carry, tone, freq=f,
                                        n_long=2, n_short=2)
        assert len(res["equity"]) > 0
