"""Integration tests for the orchestrator — every external call is mocked."""
import numpy as np
import pandas as pd
import pytest
import screener
import factors
from screener import run_screen, top_n, DEFAULT_UNIVERSE
from tests.conftest import make_prices

UNIV = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]

FUND = {t: {"pe": 10 + i * 5, "pb": 1 + i, "ev_ebitda": 6 + i * 3,
            "roe": 0.30 - i * 0.04, "margin": 0.25 - i * 0.03,
            "d2e": 20 + i * 30}
        for i, t in enumerate(UNIV)}


@pytest.fixture
def mock_all(monkeypatch):
    prices = make_prices(UNIV, n_days=600, seed=5)
    monkeypatch.setattr(screener, "download_prices", lambda u, start=None: prices)
    monkeypatch.setattr(factors, "_fundamentals", lambda t: FUND[t])
    monkeypatch.setattr(screener, "sentiment_scores",
                        lambda u: pd.Series(0.0, index=u))
    monkeypatch.setattr(screener, "detect_regime",
                        lambda *a, **k: {"regime": "calm", "p_stress": 0.2,
                                         "p_calm": 0.8, "method": "test"})
    monkeypatch.setattr(screener, "ml_rank_scores",
                        lambda p: pd.Series(0.0, index=p.columns))
    return prices


def test_run_screen_output_shape(mock_all):
    df, regime = run_screen(universe=UNIV)
    assert "SCORE" in df.columns
    assert set(df.index) == set(UNIV)
    assert regime["regime"] == "calm"


def test_run_screen_is_sorted(mock_all):
    df, _ = run_screen(universe=UNIV)
    assert list(df["SCORE"]) == sorted(df["SCORE"], reverse=True)


def test_run_screen_scores_finite(mock_all):
    df, _ = run_screen(universe=UNIV)
    for c in ["value", "quality", "momentum", "network", "SCORE"]:
        assert np.isfinite(df[c]).all()


def test_top_n_count(mock_all):
    df, _ = run_screen(universe=UNIV)
    picks = top_n(df, 3)
    assert len(picks) == 3
    assert picks == list(df.head(3).index)


def test_top_n_one_per_community(mock_all):
    df, _ = run_screen(universe=UNIV)
    picks = top_n(df, 3, one_per_community=True)
    comms = [df.loc[t, "community"] for t in picks]
    assert len(comms) == len(set(comms))     # all distinct communities


def test_regime_changes_weights_effect(monkeypatch, mock_all):
    """Calm vs stress regime should generally reorder the rankings."""
    df_calm, _ = run_screen(universe=UNIV)
    monkeypatch.setattr(screener, "detect_regime",
                        lambda *a, **k: {"regime": "stress", "p_stress": 0.9,
                                         "p_calm": 0.1, "method": "test"})
    df_stress, _ = run_screen(universe=UNIV)
    # scores should differ because factor weights changed
    assert not np.allclose(df_calm["SCORE"].reindex(UNIV).values,
                           df_stress["SCORE"].reindex(UNIV).values)


def test_default_universe_nonempty():
    assert len(DEFAULT_UNIVERSE) > 10
