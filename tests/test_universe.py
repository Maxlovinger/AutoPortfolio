"""Offline tests for universe construction + eligibility filtering."""
import numpy as np
import pandas as pd
import pytest
import universe as U


def test_clean_ticker():
    assert U.clean_ticker("BRK.B") == "BRK-B"
    assert U.clean_ticker(" aapl ") == "AAPL"


def test_fetch_candidates_monkeypatched(monkeypatch):
    def fake_read(url):
        return pd.DataFrame({"ticker": ["AAA", "BBB"], "name": ["A", "B"],
                             "sector": ["Tech", "Fin"]})
    monkeypatch.setattr(U, "_read_wiki", fake_read)
    cand = U.fetch_candidates()
    assert set(cand["ticker"]) == {"AAA", "BBB"}      # deduped across 3 pages
    assert {"tier", "index"} <= set(cand.columns)


def _mk(cols, n=600):
    idx = pd.bdate_range("2019-01-01", periods=n)
    rng = np.random.default_rng(0)
    close = pd.DataFrame({c: 50 + np.cumsum(rng.normal(0, 0.5, n)) for c in cols},
                         index=idx)
    volume = pd.DataFrame({c: rng.integers(1_000_000, 5_000_000, n) for c in cols},
                          index=idx)
    return close, volume


def test_eligibility_liquidity_and_price():
    close, volume = _mk(["LIQ", "PENNY", "ILLIQ"])
    close["PENNY"] = 1.5                        # below price floor
    volume["ILLIQ"] = 10                        # tiny volume -> low ADV
    m = U.apply_eligibility(close, volume, min_price=3.0, min_adv=1_000_000,
                            min_days=504)
    assert m.loc["LIQ", "eligible"]
    assert not m.loc["PENNY", "eligible"]
    assert not m.loc["ILLIQ", "eligible"]


def test_eligibility_history_floor():
    close, volume = _mk(["SHORT"], n=100)       # too little history
    m = U.apply_eligibility(close, volume, min_days=504)
    assert not m.loc["SHORT", "eligible"]


def test_build_universe_monkeypatched(monkeypatch, tmp_path):
    cand = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"],
                         "name": ["A", "B", "C"], "sector": ["T", "F", "H"],
                         "tier": ["small", "mid", "small"],
                         "index": ["SP600", "SP400", "SP600"]})
    close, volume = _mk(["AAA", "BBB", "CCC"])
    monkeypatch.setattr(U, "fetch_candidates", lambda: cand)
    monkeypatch.setattr(U, "download_price_volume", lambda t, period="5y": (close, volume))
    prefix = str(tmp_path / "uni")
    tickers = U.build_universe(top_n=2, out_prefix=prefix, verbose=False)
    assert len(tickers) == 2
    assert U.load_universe(f"{prefix}_tickers.txt") == tickers
