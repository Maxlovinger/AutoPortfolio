"""
Offline tests for the Tier-1 foreign-equity gate (foreign_data + foreign_gate).

No network: the ETF download is exercised only through its cache path, and every
analysis function is fed a synthetic monthly panel with KNOWN correlations so the
gate arithmetic (threshold rule, tail correlation, Markowitz weights, shared-tail)
is exactly checkable.
"""
import numpy as np
import pandas as pd
import pytest

import foreign_data as fd
import foreign_gate as fg
from utils import MONTH_END


# --- synthetic panel -------------------------------------------------------
def make_panel(n=150, seed=0):
    """
    Factor-built monthly panel with designed correlations:
      equity  : the market factor (positive mean)
      EAFE    : ~0.9 corr to equity   (developed, redundant)
      EM      : ~0.4 corr, HIGHER Sharpe (the diversifier)
      carry   : ~0.2 corr to equity, low vol
      Mexico  : an EM country, loaded on carry's factor (shared-tail candidate)
      EAFE_hedged : EAFE minus its FX leg -> lower vol
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-31", periods=n, freq=MONTH_END)
    eq = rng.normal(0.010, 0.040, n)                    # market factor
    fx = rng.normal(0.0, 0.030, n)                      # a currency factor
    carry_f = rng.normal(0.003, 0.015, n)
    eafe_h = 0.9 * eq + rng.normal(0, 0.010, n)         # hedged base (no FX)
    eafe = eafe_h + 0.6 * fx                            # unhedged = base + FX leg
    em = 0.4 * eq + 0.02 + rng.normal(0, 0.045, n)      # higher mean, lower corr
    carry = 0.2 * eq + carry_f
    mexico = 0.3 * eq + 0.6 * carry_f + rng.normal(0, 0.03, n)  # tied to carry
    df = pd.DataFrame({
        "equity": eq, "carry": carry, "EAFE": eafe, "EAFE_hedged": eafe_h,
        "EM": em, "Mexico": mexico,
    }, index=idx)
    return df


# --- foreign_data universe integrity ---------------------------------------
def test_all_tickers_unique_and_spy_once():
    ts = fd.all_tickers()
    assert len(ts) == len(set(ts))
    assert ts.count("SPY") == 1
    # every friendly name maps to exactly one ticker
    assert len(fd.ALL) == len(set(fd.ALL.values()))


def test_em_carry_map_points_at_real_country_etfs():
    # each EM carry currency maps to a name that IS a defined EM country ETF
    for ccy, name in fd.EM_CARRY_ETF.items():
        assert name in fd.EM_COUNTRY, f"{ccy}->{name} not an EM country ETF"
    # CZK/HUF deliberately absent (no liquid single-country US ETF)
    assert "CZK" not in fd.EM_CARRY_ETF and "HUF" not in fd.EM_CARRY_ETF


def test_hedge_pairs_reference_known_names():
    for unh, hed in fd.HEDGE_PAIRS:
        assert unh in fd.ALL and hed in fd.ALL


def test_monthly_total_return_from_prices():
    idx = pd.bdate_range("2020-01-01", periods=90)
    px = pd.DataFrame({"X": np.linspace(100, 118, 90)}, index=idx)
    r = fd._monthly_total_return(px)
    # month-over-month last-price change, first row NaN
    assert r["X"].iloc[0] != r["X"].iloc[0] or np.isnan(r["X"].iloc[0])
    assert (r["X"].dropna() > 0).all()          # monotone up -> positive returns


def test_download_returns_uses_cache(tmp_path, monkeypatch):
    # populate a fake cache and confirm download_returns reads it (no network)
    cache = tmp_path / "foreign_prices.pkl"
    idx = pd.date_range("2010-01-31", periods=40, freq=MONTH_END)
    fake = pd.DataFrame({"US": np.random.randn(40) * 0.01,
                         "EM": np.random.randn(40) * 0.02}, index=idx)
    fake.to_pickle(cache)
    monkeypatch.setattr(fd, "CACHE", str(cache))
    out = fd.download_returns(start="2010-01-01")
    assert list(out.columns) == ["US", "EM"]
    # start filter applied
    out2 = fd.download_returns(start="2011-01-01")
    assert out2.index[0] >= pd.Timestamp("2011-01-01")


# --- gate: correlation & threshold -----------------------------------------
def test_gate_table_developed_is_high_corr_em_is_low():
    panel = make_panel(seed=1)
    tab = fg.gate_table(panel)
    assert tab.loc["EAFE", "corr_equity"] > 0.75      # designed redundant
    assert tab.loc["EM", "corr_equity"] < 0.6         # designed diversifier
    assert tab.loc["EAFE", "group"] == "developed"
    assert tab.loc["EM", "group"] == "em"


def test_gate_threshold_rule_matches_formula():
    panel = make_panel(seed=2)
    tab = fg.gate_table(panel)
    sh_eq = fg._sharpe(panel["equity"])
    for name in tab.index:
        corr = tab.loc[name, "corr_equity"]
        sh_f = tab.loc[name, "sharpe"]
        expected = bool((sh_f > corr * sh_eq) and (sh_f > 0))
        assert bool(tab.loc[name, "earns_weight"]) == expected
        assert tab.loc[name, "threshold"] == pytest.approx(corr * sh_eq)


def test_tail_corr_uses_only_worst_months():
    # anchor with a clear tail; a,b perfectly correlated only in the tail
    n = 60
    idx = pd.date_range("2010-01-31", periods=n, freq=MONTH_END)
    anchor = pd.Series(np.linspace(-0.2, 0.2, n), index=idx)   # sorted
    a = anchor.copy()
    b = anchor.copy()
    # scramble the calm (upper) part so full corr < tail corr
    b.iloc[20:] = np.random.default_rng(0).normal(0, 0.1, n - 20)
    tc = fg._tail_corr(a, b, anchor, q=0.2)
    assert tc == pytest.approx(1.0, abs=1e-6)     # tail is the sorted, matching part


# --- Markowitz add ---------------------------------------------------------
def test_markowitz_add_weights_valid_and_baseline_present():
    panel = make_panel(seed=3)
    res = fg.markowitz_add(panel, "EM")
    w = np.array(list(res["weights"].values()))
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-9).all()                      # long-only
    assert set(res["weights"]) == {"equity", "carry", "EM"}
    assert np.isfinite(res["sharpe"]) and np.isfinite(res["base_sharpe"])


def test_markowitz_add_high_sharpe_asset_gets_weight():
    # EM built with higher mean & low corr -> tangency should hold some of it
    panel = make_panel(seed=4)
    res = fg.markowitz_add(panel, "EM")
    assert res["foreign_weight"] > 0.01


# --- EM shared tail --------------------------------------------------------
def test_em_shared_tail_detects_carry_linkage():
    panel = make_panel(seed=5)
    em = fg.em_shared_tail(panel)
    # only mapped currencies whose ETF exists in the panel appear (Mexico here)
    assert any("Mexico" in i for i in em.index)
    row = em.loc[[i for i in em.index if "Mexico" in i][0]]
    # Mexico was loaded on carry's factor -> positive correlation to carry
    assert row["corr_carry"] > 0.2


# --- hedge effect ----------------------------------------------------------
def test_hedge_effect_hedged_has_lower_vol():
    panel = make_panel(seed=6)
    he = fg.hedge_effect(panel)
    assert "EAFE" in he.index
    # FX leg stripped -> hedged vol < unhedged vol by construction
    assert he.loc["EAFE", "vol_hedged"] < he.loc["EAFE", "vol_unhedged"]


# --- figure smoke ----------------------------------------------------------
def test_make_figure_writes_png(tmp_path, monkeypatch):
    panel = make_panel(seed=7)
    monkeypatch.setattr(fg, "OUT", str(tmp_path))
    p = fg.make_figure(panel)
    import os
    assert os.path.exists(p) and p.endswith(".png")
