"""
Tests for risk-based allocators. Offline, synthetic returns with known structure.
"""
import numpy as np
import pandas as pd
import pytest

from allocators import (equal_weight, inverse_vol, min_variance, erc,
                        max_diversification, hrp, allocate, ALLOCATORS)


@pytest.fixture
def rets():
    rng = np.random.default_rng(7)
    n = 400
    # 4 assets with clearly different vols; last one very calm
    data = {
        "HI": rng.normal(0, 0.03, n),
        "MED1": rng.normal(0, 0.015, n),
        "MED2": rng.normal(0, 0.015, n),
        "LO": rng.normal(0, 0.004, n),
    }
    return pd.DataFrame(data)


@pytest.mark.parametrize("name,fn", list(ALLOCATORS.items()))
def test_weights_sum_to_one_and_nonneg(name, fn, rets):
    w = fn(rets)
    assert abs(w.sum() - 1) < 1e-6, name
    assert (w >= -1e-9).all(), name
    assert set(w.index) == set(rets.columns), name


def test_inverse_vol_favors_calm_asset(rets):
    w = inverse_vol(rets)
    assert w["LO"] > w["HI"]                 # lower vol -> bigger weight


def test_min_variance_tilts_to_low_vol(rets):
    w = min_variance(rets)
    assert w["LO"] > w["HI"]


def test_min_variance_respects_cap(rets):
    w = min_variance(rets, cap=0.4)
    assert (w <= 0.4 + 1e-6).all()


def test_erc_risk_contributions_roughly_equal(rets):
    from allocators import _cov
    w = erc(rets).values
    Sig = _cov(rets)
    rc = w * (Sig @ w)
    # risk contributions should be far more equal than 1/N's would be
    assert rc.std() / rc.mean() < 0.25


def test_max_diversification_beats_naive_dr(rets):
    from allocators import _cov
    Sig = _cov(rets)
    vol = np.sqrt(np.diag(Sig))

    def dr(w):
        return (w @ vol) / np.sqrt(w @ Sig @ w)

    w_md = max_diversification(rets).values
    w_eq = np.repeat(0.25, 4)
    assert dr(w_md) >= dr(w_eq) - 1e-9


def test_hrp_runs_and_diversifies(rets):
    w = hrp(rets)
    assert abs(w.sum() - 1) < 1e-6
    assert (w > 0).all()                      # HRP spreads across all names
    assert w["LO"] > w["HI"]                  # still risk-aware


def test_allocate_single_name():
    r = pd.DataFrame({"A": np.random.default_rng(1).normal(0, 0.01, 300)})
    w = allocate(r, "hrp")
    assert w.to_dict() == {"A": 1.0}


def test_allocate_empty():
    w = allocate(pd.DataFrame(), "equal")
    assert w.empty
