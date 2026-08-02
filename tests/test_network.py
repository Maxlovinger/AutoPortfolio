"""Tests for the (C) correlation-network diversifier signal."""
import numpy as np
import pandas as pd
import pytest
from network_model import build_correlation_graph, network_scores, community_labels


def test_graph_has_edges_for_correlated(prices):
    G = build_correlation_graph(prices, threshold=0.3)
    assert G.number_of_nodes() == prices.shape[1]
    assert G.number_of_edges() >= 1        # shared market factor -> correlation


def test_network_scores_zero_mean_finite(prices):
    s = network_scores(prices, threshold=0.3)
    assert np.isfinite(s).all()
    assert abs(s.mean()) < 1e-6            # z-scored


def test_diversifier_gets_higher_score():
    # 3 tightly-linked names + 1 independent name
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.normal(0, 1, 400)) + 100
    df = pd.DataFrame({
        "A": base + rng.normal(0, 0.2, 400),
        "B": base + rng.normal(0, 0.2, 400),
        "C": base + rng.normal(0, 0.2, 400),
        "LONER": np.cumsum(rng.normal(0, 1, 400)) + 100,
    }, index=idx)
    s = network_scores(df, threshold=0.4)
    assert s["LONER"] == s.max()           # least central -> best diversifier


def test_disconnected_graph_all_zero():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({c: np.cumsum(rng.normal(0, 1, 300)) + 100
                       for c in ["A", "B", "C"]}, index=idx)
    s = network_scores(df, threshold=0.99)   # no edges clear the bar
    assert np.allclose(s.values, 0.0)


def test_community_labels_cover_all_nodes(prices):
    labels = community_labels(prices, threshold=0.3)
    assert set(labels.keys()) == set(prices.columns)
