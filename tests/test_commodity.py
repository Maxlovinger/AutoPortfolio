"""
Offline tests for the Tier-1 commodity gate (commodity_data + commodity_gate).

No network: the ETF download is exercised via its cache path, and the trend book
+ gate arithmetic run on synthetic monthly panels with KNOWN structure so the
no-lookahead property, crisis-alpha number, and threshold rule are checkable.
"""
import numpy as np
import pandas as pd
import pytest

import commodity_data as cd
import commodity_gate as cg
from utils import MONTH_END


# --- synthetic panels ------------------------------------------------------
def make_comm_rets(n=120, seed=0):
    """Monthly commodity returns for the trend book names, with a clear
    persistent up-trend in Gold and down-trend in Oil (so momentum has signal)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-31", periods=n, freq=MONTH_END)
    data = {}
    for nm in cd.TREND_BOOK:
        data[nm] = rng.normal(0.0, 0.05, n)
    data["Gold"] = rng.normal(0.012, 0.04, n)      # steady up-trend
    data["Oil"] = rng.normal(-0.010, 0.05, n)      # steady down-trend
    return pd.DataFrame(data, index=idx)


def make_panel(n=140, seed=0):
    """equity + carry + a passive commodity + a crisis-hedging trend stream."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-31", periods=n, freq=MONTH_END)
    eq = rng.normal(0.010, 0.040, n)
    carry = 0.2 * eq + rng.normal(0.003, 0.015, n)
    passive = 0.3 * eq + rng.normal(0.002, 0.05, n)          # mild corr to eq
    trend = -0.5 * eq + rng.normal(0.008, 0.03, n)           # NEGATIVE corr (hedge)
    return pd.DataFrame({"equity": eq, "carry": carry,
                         "Commodities": passive, "comm_trend": trend}, index=idx)


# --- data universe integrity -----------------------------------------------
def test_all_tickers_unique_spy_once():
    ts = cd.all_tickers()
    assert len(ts) == len(set(ts))
    assert ts.count("SPY") == 1
    assert len(cd.ALL) == len(set(cd.ALL.values()))


def test_trend_book_names_are_defined_commodities():
    for nm in cd.TREND_BOOK:
        assert nm in cd.ALL and cd.ALL[nm] not in ("SPY",)


def test_download_returns_uses_cache(tmp_path, monkeypatch):
    cache = tmp_path / "commodity_prices.pkl"
    idx = pd.date_range("2008-01-31", periods=30, freq=MONTH_END)
    fake = pd.DataFrame({"Commodities": np.random.randn(30) * 0.02,
                         "Gold": np.random.randn(30) * 0.03}, index=idx)
    fake.to_pickle(cache)
    monkeypatch.setattr(cd, "CACHE", str(cache))
    out = cd.download_returns(start="2008-01-01")
    assert list(out.columns) == ["Commodities", "Gold"]
    assert cd.download_returns(start="2009-01-01").index[0] >= pd.Timestamp("2009-01-01")


# --- trend strategy --------------------------------------------------------
def test_trend_signal_is_lagged_no_lookahead():
    # changing ONLY the contemporaneous month must not change that month's position
    R = make_comm_rets(seed=1)
    strat = cg.trend_strategy(R)
    R2 = R.copy()
    t = R.index[60]
    R2.loc[t] = R2.loc[t] + 5.0               # violent contemporaneous shock
    strat2 = cg.trend_strategy(R2)
    # month t's strategy return = position_t (fixed, from t-1 info) * R_t, so it
    # scales with R_t; but months BEFORE t must be identical (no future leak)
    before = R.index[:60]
    pd.testing.assert_series_equal(strat.loc[before].dropna(),
                                   strat2.loc[before].dropna())


def test_trend_goes_long_persistent_uptrend():
    # Gold trends up -> after 12m the position should be long, profiting on average
    R = make_comm_rets(seed=2)
    only_gold = cg.trend_strategy(R, names=["Gold"])
    assert only_gold.dropna().mean() > 0     # long an up-trend earns


def test_trend_profits_shorting_a_downtrend():
    R = make_comm_rets(seed=3)
    only_oil = cg.trend_strategy(R, names=["Oil"])   # persistent down-trend
    # momentum flips short and earns as it keeps falling
    assert only_oil.dropna().mean() > 0


def test_trend_empty_when_no_names_present():
    R = pd.DataFrame({"Zzz": [0.01, 0.02]},
                     index=pd.date_range("2020-01-31", periods=2, freq=MONTH_END))
    assert cg.trend_strategy(R).empty


# --- crisis-alpha metric ---------------------------------------------------
def test_tail_mean_uses_worst_anchor_months():
    n = 50
    idx = pd.date_range("2010-01-31", periods=n, freq=MONTH_END)
    anchor = pd.Series(np.linspace(-0.3, 0.3, n), index=idx)   # sorted
    a = pd.Series(np.where(anchor < 0, 0.05, -0.05), index=idx)  # +5% when anchor<0
    tm = cg._tail_mean(a, anchor, q=0.2)
    assert tm == pytest.approx(0.05)          # worst months are the anchor<0 ones


def test_strategy_gate_flags_trend_crisis_hedge():
    panel = make_panel(seed=4)
    sg = cg.strategy_gate(panel)
    assert "comm_trend" in sg.index
    # trend built NEGATIVELY correlated to equity -> negative corr & positive
    # crisis mean (rises when equity falls)
    assert sg.loc["comm_trend", "corr_equity"] < 0
    assert sg.loc["comm_trend", "crisis_mean_ret"] > 0


# --- passive gate table ----------------------------------------------------
def test_gate_table_threshold_rule():
    # build a small panel with named commodity columns from a real group
    rng = np.random.default_rng(5)
    n = 130
    idx = pd.date_range("2008-01-31", periods=n, freq=MONTH_END)
    eq = rng.normal(0.01, 0.04, n)
    panel = pd.DataFrame({
        "equity": eq, "carry": 0.2 * eq + rng.normal(0.003, 0.015, n),
        "Gold": 0.1 * eq + rng.normal(0.004, 0.05, n),
        "Oil": 0.5 * eq + rng.normal(-0.002, 0.06, n),
    }, index=idx)
    tab = cg.gate_table(panel)
    sh_eq = cg._sharpe(panel["equity"])
    for name in tab.index:
        corr, sh_f = tab.loc[name, "corr_equity"], tab.loc[name, "sharpe"]
        assert bool(tab.loc[name, "earns_weight"]) == bool((sh_f > corr * sh_eq)
                                                            and (sh_f > 0))
        assert tab.loc[name, "threshold"] == pytest.approx(corr * sh_eq)


# --- Markowitz -------------------------------------------------------------
def test_markowitz_add_weights_valid():
    panel = make_panel(seed=6)
    res = cg.markowitz_add(panel, "comm_trend")
    w = np.array(list(res["weights"].values()))
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-9).all()
    # a negatively-correlated positive-return asset should earn real weight
    assert res["foreign_weight"] > 0.05


# --- figure smoke ----------------------------------------------------------
def test_make_figure_writes_png(tmp_path, monkeypatch):
    panel = make_panel(seed=7)
    monkeypatch.setattr(cg, "OUT", str(tmp_path))
    p = cg.make_figure(panel)
    import os
    assert os.path.exists(p) and p.endswith(".png")
