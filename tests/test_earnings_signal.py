"""
Offline tests for earnings_signal.py — the PEAD (earnings-surprise) tilt.
Synthetic sparse surprise panels; checks the recent-surprise lookup, no
look-ahead, the tilt direction, cap, tone blending, and the harness.
"""
import numpy as np
import pandas as pd
import pytest

import earnings_signal as ea

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def make_prices(n=900, seed=0, beats_help=False):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    out = {}
    for t in TICKERS:
        drift = 0.0006 if (beats_help and t in ("AAA", "BBB")) else \
                (-0.0004 if (beats_help and t in ("DDD", "EEE")) else 0.0)
        out[t] = np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    return pd.DataFrame(out, index=idx)


def make_surprises(beats=("AAA", "BBB"), misses=("DDD", "EEE")):
    """Sparse quarterly surprise panel: beats +8%, misses -8%, others ~0."""
    dates = pd.date_range("2018-02-15", "2021-02-15", freq="91D")
    data = {}
    for t in TICKERS:
        v = 8.0 if t in beats else (-8.0 if t in misses else 0.5)
        data[t] = np.full(len(dates), v)
    return pd.DataFrame(data, index=dates)


# --- recent_surprise -------------------------------------------------------
def test_recent_surprise_picks_latest_in_window():
    sp = make_surprises()
    t = pd.Timestamp("2019-01-01")
    s = ea.recent_surprise(sp, TICKERS, t, window_days=120)
    assert s["AAA"] == pytest.approx(8.0)
    assert s["DDD"] == pytest.approx(-8.0)


def test_recent_surprise_zero_when_no_recent_earnings():
    sp = make_surprises()
    t = pd.Timestamp("2019-01-01")
    s = ea.recent_surprise(sp, TICKERS, t, window_days=10)   # nothing in 10d
    assert (s == 0.0).all()


def test_recent_surprise_no_lookahead():
    sp = make_surprises()
    # earnings ON/after t must not be visible (strictly earlier than t)
    t = sp.index[3]                                  # exactly an earnings date
    s = ea.recent_surprise(sp, TICKERS, t, window_days=120)
    # the surprise at t itself is excluded; only the prior one (same value here)
    prior = sp.loc[sp.index < t]
    assert not prior.empty                            # there is an earlier one
    s2 = ea.recent_surprise(sp, TICKERS, t, window_days=5)   # only same-day -> none
    assert (s2 == 0.0).all()


# --- tilt ------------------------------------------------------------------
def test_tilt_overweights_recent_beats():
    prices = make_prices()
    sp = make_surprises()
    window = prices.loc[:"2019-01-01"]
    w = ea.earnings_tilt_weight(window, TICKERS, surprise_panel=sp,
                                window_days=120, lam=0.5, cap=0.5)
    assert w["AAA"] > w["CCC"] > w["DDD"]
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_tilt_equal_when_no_panel():
    prices = make_prices()
    w = ea.earnings_tilt_weight(prices.iloc[:300], TICKERS, surprise_panel=None)
    assert np.allclose(w.values, 1.0 / len(TICKERS))


def test_tilt_equal_when_no_recent_earnings():
    prices = make_prices()
    sp = make_surprises()
    window = prices.loc[:"2019-01-01"]
    w = ea.earnings_tilt_weight(window, TICKERS, surprise_panel=sp, window_days=5)
    assert np.allclose(w.values, 1.0 / len(TICKERS))    # nothing recent -> equal


def test_tilt_respects_cap():
    prices = make_prices()
    sp = make_surprises(beats=("AAA",), misses=("BBB", "CCC", "DDD", "EEE"))
    window = prices.loc[:"2019-01-01"]
    w = ea.earnings_tilt_weight(window, TICKERS, surprise_panel=sp,
                                window_days=120, lam=5.0, cap=0.30)
    assert w.max() <= 0.30 + 1e-9


def test_tilt_blends_tone():
    prices = make_prices()
    sp = make_surprises(beats=(), misses=())            # flat surprise -> z=0
    # tone favors EEE strongly; with surprise flat, tone should drive the tilt
    m_idx = pd.date_range("2018-01-31", "2020-06-30", freq="M")
    tone = pd.DataFrame(0.0, index=m_idx, columns=TICKERS)
    tone["EEE"] = 0.9
    window = prices.loc[:"2019-06-01"]
    w = ea.earnings_tilt_weight(window, TICKERS, surprise_panel=sp,
                                window_days=200, lam=0.5, cap=0.5,
                                tone_panel=tone, tone_w=1.0)
    assert w["EEE"] > 1.0 / len(TICKERS)


# --- harness ---------------------------------------------------------------
def test_run_compare_fixed_shapes_and_null_ish():
    prices = make_prices(beats_help=True)
    sp = make_surprises()
    out = ea.run_compare_fixed(prices, sp, lookback=252, rebalance=63,
                               train_end="2020-06-30", lam=0.5, cap=0.5)
    assert set(out) == {"equal-weight", "earnings-tilt"}
    for k in out:
        assert "full" in out[k] and "test" in out[k]


def test_tilt_beats_equal_when_beats_outperform():
    prices = make_prices(seed=2, beats_help=True)
    sp = make_surprises()
    out = ea.run_compare_fixed(prices, sp, lookback=252, rebalance=63,
                               train_end="2020-06-30", lam=1.0, cap=0.5,
                               window_days=120)
    assert out["earnings-tilt"]["full"]["sharpe"] > out["equal-weight"]["full"]["sharpe"]
