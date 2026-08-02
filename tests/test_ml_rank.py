"""Tests for the (E) ML learning-to-rank model."""
import numpy as np
import pandas as pd
import pytest
from ml_rank import (
    _features_for, build_panel, ml_rank_scores, _relevance_labels,
)
from tests.conftest import make_prices


def test_features_columns():
    px = make_prices(["AAA"], n_days=400)["AAA"]
    f = _features_for(px)
    for c in ["mom_21", "mom_252", "vol_21", "revert", "dist_high"]:
        assert c in f.columns


def test_build_panel_has_target_and_ids():
    prices = make_prices(["AAA", "BBB", "CCC"], n_days=500)
    panel = build_panel(prices)
    assert {"fwd_ret", "ticker", "date"} <= set(panel.columns)
    assert panel["ticker"].nunique() == 3


def test_relevance_labels_constant_is_zero():
    lab = _relevance_labels(pd.Series([0.1, 0.1, 0.1, 0.1]))
    assert (lab == 0).all()


def test_relevance_labels_varied_has_spread():
    lab = _relevance_labels(pd.Series(np.linspace(-0.1, 0.1, 20)))
    assert lab.nunique() > 1


def test_ml_rank_scores_finite_and_indexed():
    prices = make_prices(["AAA", "BBB", "CCC", "DDD", "EEE"],
                         n_days=600, seed=7)
    s = ml_rank_scores(prices)
    assert list(s.index) == list(prices.columns)
    assert np.isfinite(s).all()


def test_ml_rank_insufficient_data_returns_zeros():
    prices = make_prices(["AAA", "BBB"], n_days=120)
    s = ml_rank_scores(prices)
    assert (s == 0).all()


def test_ml_rank_empty_universe():
    s = ml_rank_scores(pd.DataFrame())
    assert len(s) == 0
