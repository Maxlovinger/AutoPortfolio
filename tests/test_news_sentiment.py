"""
Offline tests for news_sentiment.py. The FinBERT path is exercised only if the
model is actually present (guarded skip); the fallback chain (finbert -> vader
-> bag-of-words) is tested deterministically via monkeypatching so it never
needs the network.
"""
import numpy as np
import pytest

import news_sentiment as ns


# --- fallback chain --------------------------------------------------------
def test_bagofwords_fallback_signs(monkeypatch):
    # force both finbert and vader unavailable -> bag-of-words
    monkeypatch.setattr(ns, "_get_finbert", lambda: (_ for _ in ()).throw(RuntimeError))
    monkeypatch.setattr(ns, "_get_vader", lambda: False)
    s = ns.score_texts(["record profit growth strong beat",
                        "fraud lawsuit probe plunge loss"], prefer="finbert")
    assert s[0] > 0 > s[1]


def test_empty_returns_neutral():
    assert ns.aggregate([]) == 0.0
    assert ns.aggregate(None) == 0.0
    assert len(ns.score_texts([])) == 0


def test_vader_used_when_finbert_unavailable(monkeypatch):
    class FakeVader:
        def polarity_scores(self, t):
            return {"compound": 0.5 if "good" in t else -0.5}
    monkeypatch.setattr(ns, "_get_finbert", lambda: (_ for _ in ()).throw(RuntimeError))
    monkeypatch.setattr(ns, "_get_vader", lambda: FakeVader())
    s = ns.score_texts(["good news", "bad news"], prefer="finbert")
    assert s[0] == pytest.approx(0.5) and s[1] == pytest.approx(-0.5)


def test_aggregate_is_mean(monkeypatch):
    monkeypatch.setattr(ns, "score_texts", lambda texts, prefer="finbert":
                        np.array([0.2, 0.4, 0.6]))
    assert ns.aggregate(["a", "b", "c"]) == pytest.approx(0.4)


def test_available_backend_reports_fallback(monkeypatch):
    monkeypatch.setattr(ns, "_get_finbert", lambda: (_ for _ in ()).throw(RuntimeError))
    monkeypatch.setattr(ns, "_get_vader", lambda: object())
    assert ns.available_backend("finbert") == "vader"
    monkeypatch.setattr(ns, "_get_vader", lambda: False)
    assert ns.available_backend("finbert") == "bagofwords"


# --- real FinBERT (only if the model is available locally) -----------------
def _finbert_ready():
    try:
        ns._get_finbert()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _finbert_ready(), reason="FinBERT model not available")
def test_finbert_orders_financial_text():
    s = ns.score_texts([
        "Company beats earnings expectations and raises guidance",
        "Firm faces fraud probe, shares plunge on lawsuit",
    ], prefer="finbert")
    assert s[0] > 0.5              # clearly positive
    assert s[1] < -0.5             # clearly negative
