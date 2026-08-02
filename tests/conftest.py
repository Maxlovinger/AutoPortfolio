"""Shared fixtures + synthetic data generators. All tests run OFFLINE."""
import numpy as np
import pandas as pd
import pytest


def _gbm(n_days, mu=0.08, sigma=0.2, s0=100.0, seed=0):
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_days)
    return s0 * np.exp(np.cumsum(shocks))


def make_prices(tickers, n_days=800, seed=0, common=0.5):
    """Synthetic price panel with a shared market factor + idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    market = _gbm(n_days, sigma=0.18, seed=seed)
    data = {}
    for i, t in enumerate(tickers):
        idio = _gbm(n_days, mu=0.05 + 0.02 * i, sigma=0.25, seed=seed + i + 1)
        px = common * market + (1 - common) * idio
        data[t] = px
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def tickers4():
    return ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture
def prices(tickers4):
    return make_prices(tickers4, n_days=800, seed=1)


@pytest.fixture
def prices_long():
    return make_prices(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
                       n_days=600, seed=2)


@pytest.fixture
def calm_returns():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0004, 0.006, 400)          # low-vol
    idx = pd.bdate_range("2020-01-01", periods=400)
    return pd.Series(r, index=idx)


@pytest.fixture
def stress_returns():
    rng = np.random.default_rng(4)
    calm = rng.normal(0.0004, 0.006, 200)
    stress = rng.normal(-0.001, 0.03, 200)      # high-vol crash tail
    idx = pd.bdate_range("2020-01-01", periods=400)
    return pd.Series(np.concatenate([calm, stress]), index=idx)
