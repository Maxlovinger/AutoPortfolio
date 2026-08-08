"""
Offline tests for equity_sentiment.py — the sentiment weight-tilt. Synthetic
prices + tone panels (no network); we check the tilt direction, no-look-ahead
(uses only months ending before the decision date), cap/normalization, graceful
fallback, and that on data where high-tone names outperform the tilt beats
equal weight.
"""
import numpy as np
import pandas as pd
import pytest

import equity_sentiment as es
from backtester import weight_equal


TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def make_prices(n=900, seed=0, tone_helps=False):
    """Daily prices. If tone_helps, AAA/BBB (which will get high tone) drift up."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    out = {}
    for i, t in enumerate(TICKERS):
        drift = 0.0
        if tone_helps and t in ("AAA", "BBB"):
            drift = 0.0006
        if tone_helps and t in ("DDD", "EEE"):
            drift = -0.0004
        out[t] = np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    return pd.DataFrame(out, index=idx)


def make_tone(prices, high=("AAA", "BBB"), low=("DDD", "EEE")):
    """Monthly tone panel: persistent high tone for `high`, low for `low`."""
    m_idx = pd.date_range(prices.index[0], prices.index[-1], freq="M")
    data = {}
    for t in TICKERS:
        v = 0.6 if t in high else (-0.6 if t in low else 0.0)
        data[t] = np.full(len(m_idx), v)
    return pd.DataFrame(data, index=m_idx)


# --- tilt direction / mechanics --------------------------------------------
def test_tilt_overweights_high_tone():
    prices = make_prices()
    tone = make_tone(prices)
    window = prices.iloc[:400]
    # cap=0.5 is feasible for 5 names (equal weight 0.2); the deployed 30-name
    # book uses cap=0.15, which rarely binds there
    w = es.sentiment_tilt_weight(window, TICKERS, tone_panel=tone, lam=0.5, cap=0.5)
    assert w["AAA"] > w["CCC"] > w["DDD"]      # high > neutral > low tone
    assert w.sum() == pytest.approx(1.0)       # normalized
    assert (w >= 0).all()                      # long-only


def test_tilt_respects_cap():
    prices = make_prices()
    tone = make_tone(prices, high=("AAA",), low=("BBB", "CCC", "DDD", "EEE"))
    window = prices.iloc[:400]
    w = es.sentiment_tilt_weight(window, TICKERS, tone_panel=tone,
                                 lam=5.0, cap=0.30)
    assert w.max() <= 0.30 + 1e-9


def test_no_tone_is_equal_weight():
    prices = make_prices()
    window = prices.iloc[:400]
    w = es.sentiment_tilt_weight(window, TICKERS, tone_panel=None)
    assert np.allclose(w.values, weight_equal(window, TICKERS).values)


def test_zero_dispersion_falls_back_to_equal():
    prices = make_prices()
    m_idx = pd.date_range(prices.index[0], prices.index[-1], freq="M")
    flat = pd.DataFrame(0.3, index=m_idx, columns=TICKERS)   # all identical tone
    window = prices.iloc[:400]
    w = es.sentiment_tilt_weight(window, TICKERS, tone_panel=flat)
    assert np.allclose(w.values, 1.0 / len(TICKERS))


# --- no look-ahead ---------------------------------------------------------
def test_tilt_uses_only_months_before_decision():
    prices = make_prices()
    # tone flips sign at a known month; weight at date t must reflect only
    # PRIOR months, never the current/future one.
    m_idx = pd.date_range(prices.index[0], prices.index[-1], freq="M")
    tone = pd.DataFrame(0.0, index=m_idx, columns=TICKERS)
    flip = m_idx[6]
    tone.loc[tone.index < flip, "AAA"] = 0.9        # positive before flip
    tone.loc[tone.index >= flip, "AAA"] = -0.9      # negative from flip on
    # decision exactly on the flip month-end: must still see the PRE-flip (+) tone
    window = prices.loc[:flip]
    w = es.sentiment_tilt_weight(window, TICKERS, tone_panel=tone, lam=0.5, cap=0.5)
    assert w["AAA"] > 1.0 / len(TICKERS)            # overweight, using prior + tone


def test_future_tone_change_does_not_affect_past_weight():
    prices = make_prices()
    m_idx = pd.date_range(prices.index[0], prices.index[-1], freq="M")
    base = pd.DataFrame(0.0, index=m_idx, columns=TICKERS)
    base["AAA"] = 0.5
    window = prices.loc[:m_idx[5]]
    w1 = es.sentiment_tilt_weight(window, TICKERS, tone_panel=base, lam=0.5)
    future = base.copy()
    future.loc[future.index >= m_idx[6], "AAA"] = -5.0   # change only the FUTURE
    w2 = es.sentiment_tilt_weight(window, TICKERS, tone_panel=future, lam=0.5)
    assert np.allclose(w1.values, w2.values)            # past weight unchanged


# --- end-to-end comparison -------------------------------------------------
def test_run_compare_shapes():
    prices = make_prices(tone_helps=True)
    tone = make_tone(prices)
    out = es.run_compare(prices, tone, lookback=252, rebalance=63,
                         train_end="2020-06-30", cap=0.5)
    assert set(out) == {"equal-weight", "sentiment-tilt"}
    for k in out:
        assert "full" in out[k] and "test" in out[k]


def test_tilt_beats_equal_when_tone_predicts():
    # engineered world: high-tone names really do outperform -> tilt should win
    prices = make_prices(seed=3, tone_helps=True)
    tone = make_tone(prices)
    out = es.run_compare(prices, tone, lookback=252, rebalance=63,
                         train_end="2020-06-30", lam=1.0, cap=0.5)
    assert out["sentiment-tilt"]["full"]["sharpe"] > out["equal-weight"]["full"]["sharpe"]


def test_tz_aware_tone_is_handled():
    # GDELT stamps tone in UTC (tz-aware); price dates are tz-naive. Comparing
    # them directly raised "Cannot compare tz-naive and tz-aware". _naive_tone
    # must normalize so the tilt works.
    prices = make_prices()
    m_idx = pd.date_range("2018-06-30", "2020-06-30", freq="M", tz="UTC")
    tone = pd.DataFrame(0.0, index=m_idx, columns=TICKERS)
    tone["AAA"] = 0.8
    assert tone.index.tz is not None
    naive = es._naive_tone(tone)
    assert naive.index.tz is None
    window = prices.iloc[:400]
    w = es.sentiment_tilt_weight(window, TICKERS, tone_panel=naive, lam=0.5, cap=0.5)
    assert w["AAA"] > 1.0 / len(TICKERS)      # overweighted, no tz crash


def test_run_compare_accepts_tz_aware_tone():
    prices = make_prices(tone_helps=True)
    m_idx = pd.date_range("2018-06-30", "2021-06-30", freq="M", tz="UTC")
    tone = make_tone(prices).copy()
    tone.index = pd.date_range("2018-01-31", periods=len(tone), freq="M", tz="UTC")
    out = es.run_compare(prices, tone, train_end="2020-06-30", cap=0.5)   # no crash
    assert set(out) == {"equal-weight", "sentiment-tilt"}


def test_ticker_names_from_universe():
    # real universe.csv should map at least some liquid tickers to names
    names = es.ticker_names(["AAPL", "NVDA", "MU"])
    assert isinstance(names, dict)
    assert all(isinstance(v, str) and v for v in names.values())
