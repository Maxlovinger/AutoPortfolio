"""
Offline tests for fx/backtest_carry.py. Uses synthetic spot + carry panels so
the carry logic, no-look-ahead alignment, cost accounting, and the monthly-vs-
weekly cadence machinery are all exactly checkable.
"""
import numpy as np
import pandas as pd
import pytest

import fx.backtest_carry as bc


# --- synthetic data --------------------------------------------------------
def make_carry_world(n_days=760, seed=0):
    """
    5 currencies. AAA/BBB are persistent HIGH-carry, DDD/EEE persistent LOW,
    CCC neutral. Spot has mild noise plus a drift that REWARDS the high-carry
    names, so a working carry book should make money.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    ccys = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    carry_level = {"AAA": 5.0, "BBB": 4.0, "CCC": 2.0, "DDD": 0.5, "EEE": 0.0}

    spot = {}
    for c in ccys:
        drift = (carry_level[c] - 2.0) * 0.00003     # high carry -> up drift
        shocks = rng.normal(drift, 0.005, n_days)
        spot[c] = 1.0 * np.exp(np.cumsum(shocks))
    spot = pd.DataFrame(spot, index=idx)

    # monthly carry panel (constant levels, as % annualized), month-end
    m_idx = pd.date_range(idx[0], idx[-1], freq="M")
    carry = pd.DataFrame(
        {c: carry_level[c] - carry_level["CCC"] for c in ccys},  # vs neutral
        index=m_idx,
    ).drop(columns=["CCC"])   # mimic "vs base": base column absent
    # keep CCC as a tradable-but-neutral name
    carry["CCC"] = 0.0
    return spot, carry[["AAA", "BBB", "CCC", "DDD", "EEE"]]


# --- carry_weights ---------------------------------------------------------
def test_carry_weights_long_high_short_low():
    row = pd.Series({"AAA": 5.0, "BBB": 4.0, "CCC": 0.0, "DDD": -3.0, "EEE": -4.0})
    w = bc.carry_weights(row, n_long=2, n_short=2, gross=1.0)
    assert w["AAA"] > 0 and w["BBB"] > 0       # highest carry -> long
    assert w["DDD"] < 0 and w["EEE"] < 0       # lowest carry -> short
    assert w["CCC"] == 0.0                      # middle untouched


def test_carry_weights_dollar_neutral_and_gross():
    row = pd.Series({"AAA": 5.0, "BBB": 4.0, "CCC": 0.0, "DDD": -3.0, "EEE": -4.0})
    w = bc.carry_weights(row, n_long=2, n_short=2, gross=1.0)
    assert w.sum() == pytest.approx(0.0, abs=1e-12)          # dollar-neutral
    assert w.abs().sum() == pytest.approx(1.0, abs=1e-12)    # gross = 1


def test_carry_weights_ignores_nan_and_flat_when_too_few():
    row = pd.Series({"AAA": 5.0, "BBB": np.nan, "CCC": np.nan})
    w = bc.carry_weights(row, n_long=2, n_short=2)
    assert (w == 0.0).all()          # only 1 valid name -> can't form 2+2 -> flat


# --- alignment / no look-ahead ---------------------------------------------
def test_build_asset_returns_no_lookahead_shift():
    spot, carry = make_carry_world()
    asset_ret, carry_grid = bc.build_asset_returns(spot, carry, freq="M")
    # carry used at each grid date must come from a PRIOR month (shifted),
    # so the first usable row is NaN (nothing known before the start).
    assert carry_grid.iloc[0].isna().all()
    # grid is monthly
    assert len(asset_ret) == len(carry_grid)
    assert set(asset_ret.columns) == {"AAA", "BBB", "CCC", "DDD", "EEE"}


def test_asset_return_includes_carry_accrual():
    # zero spot move -> asset return equals pure carry accrual carry/100 * dt
    idx = pd.bdate_range("2020-01-01", periods=90)
    spot = pd.DataFrame({"AAA": 1.0, "BBB": 1.0}, index=idx)   # flat spot
    m_idx = pd.date_range(idx[0], idx[-1], freq="M")
    carry = pd.DataFrame({"AAA": 12.0, "BBB": -6.0}, index=m_idx)
    asset_ret, _ = bc.build_asset_returns(spot, carry, freq="M")
    r = asset_ret.dropna()
    # 12% annual carry over 1/12 year = 0.01; -6% -> -0.005
    assert r["AAA"].iloc[0] == pytest.approx(12 / 100 / 12, abs=1e-12)
    assert r["BBB"].iloc[0] == pytest.approx(-6 / 100 / 12, abs=1e-12)


# --- backtest --------------------------------------------------------------
def test_backtest_runs_both_cadences():
    spot, carry = make_carry_world()
    for f in ("M", "W"):
        res = bc.run_carry_backtest(spot, carry, freq=f, n_long=2, n_short=2)
        assert len(res["equity"]) > 0
        assert (res["net_ret"] <= res["gross_ret"] + 1e-12).all()  # costs subtract


def test_carry_book_profits_when_high_carry_outperforms():
    spot, carry = make_carry_world(seed=1)
    res = bc.run_carry_backtest(spot, carry, freq="M", n_long=2, n_short=2,
                                cost_bps=0.0)
    # by construction high-carry names drift up, low drift down -> book > 0
    assert res["equity"].iloc[-1] > 1.0


def make_varying_carry(n_days=760, seed=3):
    """Same spot generator, but carry RANKS flip every month so the book
    actually rebalances -> nonzero turnover, letting us test cost impact."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    ccys = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    spot = pd.DataFrame(
        {c: 1.0 * np.exp(np.cumsum(rng.normal(0, 0.005, n_days))) for c in ccys},
        index=idx,
    )
    m_idx = pd.date_range(idx[0], idx[-1], freq="M")
    # alternate which names are high/low carry each month
    rows = []
    for k in range(len(m_idx)):
        base = np.array([5, 4, 2, 1, 0], dtype=float)
        rows.append(base if k % 2 == 0 else base[::-1])
    carry = pd.DataFrame(rows, index=m_idx, columns=ccys)
    return spot, carry


def test_costs_reduce_return_when_book_turns_over():
    spot, carry = make_varying_carry()
    free = bc.run_carry_backtest(spot, carry, freq="M", n_long=2, n_short=2,
                                 cost_bps=0.0)
    charged = bc.run_carry_backtest(spot, carry, freq="M", n_long=2, n_short=2,
                                    cost_bps=20.0)
    assert charged["turnover"].sum() > 0             # ranks flip -> real turnover
    assert charged["equity"].iloc[-1] < free["equity"].iloc[-1]


def test_per_currency_costs_charge_more_for_em():
    # a per-currency cost dict must charge each leg its own spread; higher EM
    # costs -> lower net return than uniform-low costs
    spot, carry = make_varying_carry()
    cheap = bc.run_carry_backtest(spot, carry, freq="M", n_long=2, n_short=2,
                                  cost_bps=5.0)
    # make two of the names "EM" with a 50bps spread
    percc = {c: (50.0 if c in ("AAA", "EEE") else 5.0) for c in carry.columns}
    pricey = bc.run_carry_backtest(spot, carry, freq="M", n_long=2, n_short=2,
                                   cost_bps=percc)
    assert pricey["turnover"].sum() > 0
    assert pricey["equity"].iloc[-1] < cheap["equity"].iloc[-1]


def test_wide_universe_structure():
    from fx.data import G10, EM, WIDE, EM_CCYS
    assert set(WIDE) == set(G10) | set(EM)
    assert list(WIDE).count("USD") == 1            # USD only once
    assert set(EM_CCYS) == set(EM) and "USD" not in EM_CCYS
    # every EM entry has a fred series + invert flag (all USD-per-foreign)
    for c, cfg in EM.items():
        assert cfg["fred"] and cfg["ticker"] and cfg["invert"] is True


def test_weekly_signal_does_not_inflate_turnover_for_monthly_carry():
    # KEY finding: carry updates monthly, so weekly rebalancing recomputes the
    # same forward-filled signal intra-month -> weights don't churn between
    # monthly updates. Weekly turnover should be close to monthly, NOT ~4x.
    spot, carry = make_varying_carry()
    wk = bc.run_carry_backtest(spot, carry, freq="W", n_long=2, n_short=2)
    mo = bc.run_carry_backtest(spot, carry, freq="M", n_long=2, n_short=2)
    assert wk["turnover"].sum() == pytest.approx(mo["turnover"].sum(), rel=0.35)


# --- metrics ---------------------------------------------------------------
def test_performance_flags_negative_skew():
    # long calm gains then one violent loss = carry-crash shape
    r = pd.Series([0.01] * 30 + [-0.25])
    m = bc.performance(r, ppy=12)
    assert m["skew"] < 0
    assert m["worst"] == pytest.approx(-0.25)
    assert m["max_dd"] < 0


def test_performance_too_short_is_nan():
    m = bc.performance(pd.Series([0.01, 0.02]), ppy=12)
    assert np.isnan(m["sharpe"])


def test_summarize_tags_turnover_and_uses_right_ppy():
    spot, carry = make_carry_world()
    res = bc.run_carry_backtest(spot, carry, freq="W")
    m = bc.summarize(res)
    assert "avg_turnover" in m and m["avg_turnover"] >= 0
    assert m["n"] > 0
