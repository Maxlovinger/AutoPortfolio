"""
Edge-case tests for the visualization layer. All PURE plot_* functions are
exercised with synthetic data; each must write a non-empty PNG and never raise,
including on degenerate inputs (single asset, zero edges, all-zero, NaNs, short
series). The live make_all() orchestrator is NOT called (it needs the network).
"""
import os
import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

import visualize as viz
from tests.conftest import make_prices

UNIV = ["AAA", "BBB", "CCC", "DDD"]


def _exists_nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


@pytest.fixture
def screen_df():
    rng = np.random.default_rng(0)
    cols = ["value", "quality", "momentum", "value_x_mom",
            "sentiment", "network", "ml"]
    df = pd.DataFrame(rng.normal(0, 1, (len(UNIV), len(cols))),
                      index=UNIV, columns=cols)
    df["SCORE"] = df[cols].mean(axis=1)
    df["community"] = [0, 0, 1, 1]
    return df.sort_values("SCORE", ascending=False)


def test_plot_price_history(tmp_path, prices):
    p = viz.plot_price_history(prices, str(tmp_path / "px.png"))
    assert _exists_nonempty(p)


def test_plot_correlation_heatmap(tmp_path, prices):
    p = viz.plot_correlation_heatmap(prices, str(tmp_path / "corr.png"))
    assert _exists_nonempty(p)


def test_plot_factor_scores(tmp_path, screen_df):
    p = viz.plot_factor_scores(screen_df, str(tmp_path / "f.png"))
    assert _exists_nonempty(p)


def test_plot_sentiment_mixed(tmp_path, screen_df):
    p = viz.plot_sentiment(screen_df["sentiment"], str(tmp_path / "s.png"))
    assert _exists_nonempty(p)


def test_plot_sentiment_all_zero(tmp_path):
    s = pd.Series(0.0, index=UNIV)
    p = viz.plot_sentiment(s, str(tmp_path / "s0.png"))
    assert _exists_nonempty(p)


def test_plot_network_connected(tmp_path, prices):
    p = viz.plot_network(prices, str(tmp_path / "net.png"), threshold=0.2)
    assert _exists_nonempty(p)


def test_plot_network_zero_edges(tmp_path, prices):
    # impossibly high threshold -> no edges -> must still render nodes
    p = viz.plot_network(prices, str(tmp_path / "net0.png"), threshold=1.01)
    assert _exists_nonempty(p)


def test_plot_regime_with_probs(tmp_path, stress_returns):
    from regime import regime_probabilities
    ps = regime_probabilities(stress_returns)
    p = viz.plot_regime(stress_returns, ps, str(tmp_path / "reg.png"))
    assert _exists_nonempty(p)


def test_plot_regime_empty_probs(tmp_path, calm_returns):
    # short/empty probability series must not crash the plot
    p = viz.plot_regime(calm_returns, pd.Series(dtype=float),
                        str(tmp_path / "reg2.png"))
    assert _exists_nonempty(p)


def test_plot_ml_importance(tmp_path):
    imp = pd.Series([0.3, 0.1, 0.5, 0.05, 0.0, 0.2, 0.15, 0.4],
                    index=["mom_21", "mom_63", "mom_126", "mom_252",
                           "vol_21", "vol_63", "revert", "dist_high"])
    p = viz.plot_ml_importance(imp, str(tmp_path / "imp.png"))
    assert _exists_nonempty(p)


def test_plot_ml_importance_all_zero(tmp_path):
    imp = pd.Series(0.0, index=["a", "b", "c"])
    p = viz.plot_ml_importance(imp, str(tmp_path / "imp0.png"))
    assert _exists_nonempty(p)


def test_plot_screen_scores(tmp_path, screen_df):
    p = viz.plot_screen_scores(screen_df, str(tmp_path / "sc.png"))
    assert _exists_nonempty(p)


def test_plot_efficient_frontier(tmp_path, prices):
    from data import returns_stats
    from markowitz import max_sharpe, min_variance
    mu, Sig, tickers = returns_stats(prices)
    w_ms, w_mv = max_sharpe(mu, Sig, 0.04), min_variance(Sig)
    p = viz.plot_efficient_frontier(mu, Sig, tickers, w_ms, w_mv, 0.04,
                                    str(tmp_path / "ef.png"))
    assert _exists_nonempty(p)


def test_plot_weights(tmp_path):
    p = viz.plot_weights([0.5, 0.3, 0.2], ["AAA", "BBB", "CCC"],
                         str(tmp_path / "w.png"))
    assert _exists_nonempty(p)


def test_plot_weights_single_asset(tmp_path):
    p = viz.plot_weights([1.0], ["AAA"], str(tmp_path / "w1.png"))
    assert _exists_nonempty(p)


def test_plot_ic_and_quantiles(tmp_path):
    from factor_analysis import analyze
    prices = make_prices([f"S{i}" for i in range(8)], n_days=700, seed=1)
    summary, quantiles, _ = analyze(prices, horizon=21)
    p1 = viz.plot_ic_summary(summary, str(tmp_path / "ic.png"))
    p2 = viz.plot_quantiles(quantiles, str(tmp_path / "q.png"))
    assert _exists_nonempty(p1) and _exists_nonempty(p2)


def test_plot_backtest(tmp_path):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    strat = pd.Series(rng.normal(0.0006, 0.01, 300), index=idx)
    bench = pd.Series(rng.normal(0.0004, 0.01, 300), index=idx)
    p = viz.plot_backtest(strat, bench, str(tmp_path / "bt.png"))
    assert _exists_nonempty(p)


def test_single_asset_price_and_corr(tmp_path):
    one = make_prices(["AAA"], n_days=300)
    assert _exists_nonempty(viz.plot_price_history(one, str(tmp_path / "p1.png")))
    assert _exists_nonempty(viz.plot_correlation_heatmap(one, str(tmp_path / "c1.png")))
