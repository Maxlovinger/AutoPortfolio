"""
Smoke tests for the allocation bake-off helpers (no file/network deps).
"""
import numpy as np
import pandas as pd
import pytest

from allocation_bakeoff import score_lowvol, make_score_liquid, make_weight
from tests.conftest import make_prices

UNIV = ["AAA", "BBB", "CCC", "DDD", "EEE"]


@pytest.fixture
def prices():
    return make_prices(UNIV, n_days=400, seed=3)


def test_score_lowvol_ranks_calmest_highest(prices):
    # inject an obviously calm and an obviously wild series
    p = prices.copy()
    p["AAA"] = 100.0 + np.linspace(0, 1, len(p))          # nearly flat -> low vol
    s = score_lowvol(p)
    assert s.idxmax() == "AAA"


def test_make_score_liquid_uses_adv(prices):
    adv = pd.Series({t: float(i + 1) for i, t in enumerate(UNIV)})
    s = make_score_liquid(adv)(prices)
    assert s.idxmax() == "EEE"           # highest ADV
    assert set(s.index) <= set(UNIV)


@pytest.mark.parametrize("method", ["equal", "inverse_vol", "min_var",
                                    "erc", "max_div", "hrp"])
def test_make_weight_valid_portfolio(prices, method):
    w = make_weight(method)(prices, UNIV)
    assert abs(w.sum() - 1) < 1e-6
    assert (w >= -1e-9).all()


def test_make_weight_handles_gappy_picks(prices):
    p = prices.copy()
    p.loc[p.index[-20:], "CCC"] = np.nan       # just delisted
    p.loc[p.index[:300], "DDD"] = np.nan        # recently listed
    w = make_weight("hrp")(p, UNIV)
    assert abs(w.sum() - 1) < 1e-6
