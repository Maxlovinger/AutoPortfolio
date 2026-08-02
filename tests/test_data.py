"""Offline tests for data download + return statistics."""
import numpy as np
import pandas as pd
import pytest
import data
from data import returns_stats
from tests.conftest import make_prices


def test_returns_stats_shapes(prices, tickers4):
    mu, Sig, names = returns_stats(prices)
    assert len(mu) == len(tickers4)
    assert Sig.shape == (len(tickers4), len(tickers4))
    assert names == tickers4


def test_covariance_symmetric_psd(prices):
    _, Sig, _ = returns_stats(prices)
    assert np.allclose(Sig, Sig.T)
    eig = np.linalg.eigvalsh(Sig)
    assert (eig > -1e-8).all()          # positive semi-definite


def test_annualization_reasonable(prices):
    mu, Sig, _ = returns_stats(prices)
    vols = np.sqrt(np.diag(Sig))
    assert (vols > 0.03).all() and (vols < 1.5).all()   # sane annualized vols


def test_download_prices_monkeypatched(monkeypatch):
    px = make_prices(["AAA", "BBB"], n_days=50)
    cols = pd.MultiIndex.from_product([["Close"], ["AAA", "BBB"]])
    raw = pd.DataFrame(np.column_stack([px["AAA"], px["BBB"]]),
                       index=px.index, columns=cols)
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: raw)
    out = data.download_prices(["AAA", "BBB"])
    assert list(out.columns) == ["AAA", "BBB"]
    assert not out.isna().any().any()
