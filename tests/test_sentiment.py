"""Tests for the (B) news-sentiment overlay — news feed is monkeypatched."""
import numpy as np
import pytest
import sentiment
from sentiment import sentiment_scores, _score_texts, _analyzer


HEADLINES = {
    "AAA": ["Company AAA beats earnings, shares surge to record high"],
    "BBB": ["BBB faces fraud lawsuit and analyst downgrade, shares plunge"],
    "CCC": [],   # no news -> neutral
    "DDD": ["DDD reports quarterly results in line with expectations"],
}


def test_score_empty_is_neutral():
    assert _score_texts([], _analyzer()) == 0.0


def test_score_positive_beats_negative():
    sia = _analyzer()
    pos = _score_texts(["fantastic record profit surge beat upgrade"], sia)
    neg = _score_texts(["terrible fraud lawsuit loss downgrade plunge"], sia)
    assert pos > neg


def test_fallback_lexicon_directionally_correct():
    # force fallback path (sia=None)
    pos = _score_texts(["surge record growth strong buy gain"], None)
    neg = _score_texts(["cut downgrade lawsuit weak sell loss"], None)
    assert pos > 0 > neg


def test_sentiment_scores_ranking(monkeypatch):
    monkeypatch.setattr(sentiment, "_headline_texts", lambda t: HEADLINES[t])
    s = sentiment_scores(["AAA", "BBB", "CCC", "DDD"])
    assert s["AAA"] > s["BBB"]
    assert set(s.index) == {"AAA", "BBB", "CCC", "DDD"}


def test_no_news_all_neutral(monkeypatch):
    monkeypatch.setattr(sentiment, "_headline_texts", lambda t: [])
    s = sentiment_scores(["AAA", "BBB", "CCC"])
    assert (s == 0).all()
