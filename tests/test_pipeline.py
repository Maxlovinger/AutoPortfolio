"""End-to-end pipeline test (screen -> allocate), fully mocked/offline."""
import numpy as np
import pandas as pd
import pytest
import pipeline
from tests.conftest import make_prices

PICKS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


@pytest.fixture
def mock_pipeline(monkeypatch):
    prices = make_prices(PICKS, n_days=600, seed=9)
    screen_df = pd.DataFrame(
        {"SCORE": np.linspace(2, -2, len(PICKS)), "community": range(len(PICKS))},
        index=PICKS,
    )
    monkeypatch.setattr(pipeline, "run_screen",
                        lambda *a, **k: (screen_df,
                                         {"regime": "calm", "p_stress": 0.2}))
    monkeypatch.setattr(pipeline, "download_prices", lambda picks, start=None: prices)
    return prices


def test_pipeline_runs_end_to_end(mock_pipeline, capsys):
    pipeline.TOP_N = 4
    pipeline.main()
    out = capsys.readouterr().out
    assert "Max-Sharpe portfolio" in out
    assert "Min-Variance portfolio" in out


def test_pipeline_weight_cap_enforced(mock_pipeline):
    """No single name should exceed MAX_WEIGHT in the capped max-Sharpe solve."""
    from data import returns_stats
    from scipy.optimize import minimize
    from markowitz import sharpe_ratio
    mu, Sig, tickers = returns_stats(mock_pipeline)
    n = len(tickers)
    bounds = [(0.0, pipeline.MAX_WEIGHT)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    w = minimize(lambda w: -sharpe_ratio(w, mu, Sig, 0.04),
                 np.repeat(1 / n, n), method="SLSQP",
                 bounds=bounds, constraints=cons).x
    assert w.max() <= pipeline.MAX_WEIGHT + 1e-6
    assert abs(w.sum() - 1) < 1e-6
