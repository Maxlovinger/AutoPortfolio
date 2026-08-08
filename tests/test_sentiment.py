"""Tests for the (B) equity news-sentiment overlay — news feed is monkeypatched.
Scoring detail lives in test_news_sentiment.py; here we test the per-ticker
fetch -> aggregate -> cross-sectional z-score wiring."""
import numpy as np
import pytest
import sentiment
from sentiment import sentiment_scores


HEADLINES = {
    "AAA": ["Company AAA beats earnings, shares surge to record high"],
    "BBB": ["BBB faces fraud lawsuit and analyst downgrade, shares plunge"],
    "CCC": [],   # no news -> neutral
    "DDD": ["DDD reports quarterly results in line with expectations"],
}


def test_sentiment_scores_ranking(monkeypatch):
    monkeypatch.setattr(sentiment, "_headline_texts", lambda t: HEADLINES[t])
    # force the lightweight scorer so the test never needs the FinBERT model
    s = sentiment_scores(["AAA", "BBB", "CCC", "DDD"], prefer="vader")
    assert s["AAA"] > s["BBB"]                       # good news ranks above bad
    assert set(s.index) == {"AAA", "BBB", "CCC", "DDD"}


def test_no_news_all_neutral(monkeypatch):
    monkeypatch.setattr(sentiment, "_headline_texts", lambda t: [])
    s = sentiment_scores(["AAA", "BBB", "CCC"], prefer="vader")
    assert (s == 0).all()


def test_scores_are_zscored(monkeypatch):
    monkeypatch.setattr(sentiment, "_headline_texts", lambda t: HEADLINES[t])
    s = sentiment_scores(["AAA", "BBB", "CCC", "DDD"], prefer="vader")
    # z-scored cross-section -> approximately mean zero
    assert abs(s.mean()) < 1e-9


def test_finbert_default_falls_back_gracefully(monkeypatch):
    # With the default prefer="finbert", if FinBERT is unavailable the scorer
    # must fall back (VADER/bag-of-words) and still return a full ranked series,
    # never crash. Force the FinBERT path to fail so no 440MB model is loaded.
    import news_sentiment
    monkeypatch.setattr(news_sentiment, "_get_finbert",
                        lambda: (_ for _ in ()).throw(RuntimeError))
    monkeypatch.setattr(sentiment, "_headline_texts", lambda t: HEADLINES[t])
    s = sentiment_scores(["AAA", "BBB", "CCC", "DDD"])   # default prefer="finbert"
    assert set(s.index) == {"AAA", "BBB", "CCC", "DDD"}
    assert s["AAA"] > s["BBB"]                            # fallback still ranks
