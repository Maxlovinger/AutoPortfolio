"""Tests for the (A) factor composite — fundamentals are monkeypatched."""
import numpy as np
import pandas as pd
import pytest
import factors
from factors import momentum_scores, build_factor_table, composite_score


FUND = {
    "AAA": {"pe": 8,  "pb": 1.0, "ev_ebitda": 5,  "roe": 0.30, "margin": 0.25, "d2e": 20},   # cheap+quality
    "BBB": {"pe": 40, "pb": 9.0, "ev_ebitda": 30, "roe": 0.05, "margin": 0.02, "d2e": 200},  # expensive+weak
    "CCC": {"pe": 15, "pb": 3.0, "ev_ebitda": 12, "roe": 0.15, "margin": 0.12, "d2e": 80},
    "DDD": {"pe": 22, "pb": 5.0, "ev_ebitda": 18, "roe": 0.10, "margin": 0.08, "d2e": 120},
}


@pytest.fixture(autouse=True)
def patch_fundamentals(monkeypatch):
    monkeypatch.setattr(factors, "_fundamentals", lambda t: FUND[t])


def test_momentum_rising_series_positive():
    idx = pd.bdate_range("2020-01-01", periods=300)
    up = pd.Series(np.linspace(100, 200, 300), index=idx)
    down = pd.Series(np.linspace(200, 100, 300), index=idx)
    df = pd.DataFrame({"UP": up, "DOWN": down})
    mom = momentum_scores(df)
    assert mom["UP"] > mom["DOWN"]


def test_value_score_prefers_cheap(prices):
    ft = build_factor_table(prices, ["AAA", "BBB", "CCC", "DDD"])
    assert ft.loc["AAA", "value"] > ft.loc["BBB", "value"]


def test_quality_score_prefers_profitable(prices):
    ft = build_factor_table(prices, ["AAA", "BBB", "CCC", "DDD"])
    assert ft.loc["AAA", "quality"] > ft.loc["BBB", "quality"]


def test_composite_ranks_best_first(prices):
    # Isolate fundamentals (value+quality); synthetic momentum is random noise
    # here and would otherwise swamp the deterministic ranking we're testing.
    ft = build_factor_table(prices, ["AAA", "BBB", "CCC", "DDD"])
    score = composite_score(ft, weights={"value": 1.0, "quality": 1.0})
    assert score.index[0] == "AAA"        # cheap+quality wins
    assert score.index[-1] == "BBB"       # expensive+weak loses


def test_interaction_column_present(prices):
    ft = build_factor_table(prices, ["AAA", "BBB", "CCC", "DDD"])
    assert "value_x_mom" in ft.columns


def test_missing_fundamentals_yield_zero(monkeypatch, prices):
    monkeypatch.setattr(factors, "_fundamentals",
                        lambda t: {k: np.nan for k in
                                   ["pe", "pb", "ev_ebitda", "roe", "margin", "d2e"]})
    ft = build_factor_table(prices, ["AAA", "BBB", "CCC", "DDD"])
    assert np.allclose(ft["value"].values, 0.0)
    assert np.allclose(ft["quality"].values, 0.0)


def test_single_ticker_does_not_crash(prices):
    ft = build_factor_table(prices[["AAA"]], ["AAA"])
    assert len(ft) == 1
    assert np.isfinite(composite_score(ft)).all()
