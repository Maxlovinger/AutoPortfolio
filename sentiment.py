"""
sentiment.py — (B) Alternative-data signal: news-headline sentiment.

For each ticker we pull recent news headlines (via yfinance) and score their
tone with VADER (NLTK). The per-ticker signal is the average compound
sentiment, z-scored across the universe. Recent-tone shifts often lead price.

Graceful degradation: if the lexicon or news feed is unavailable, every ticker
gets a neutral 0 and the pipeline continues.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

from utils import zscore, safe

# Tiny fallback lexicon if NLTK VADER isn't available offline.
_FALLBACK_POS = {"beat", "surge", "record", "upgrade", "growth", "strong", "buy", "gain"}
_FALLBACK_NEG = {"miss", "cut", "downgrade", "lawsuit", "probe", "weak", "sell", "loss", "fraud"}


def _analyzer():
    # PRIMARY: the maintained standalone vaderSentiment package.
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:
        pass
    # FALLBACK: NLTK's VADER.
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


def _headline_texts(ticker: str) -> list[str]:
    news = safe(lambda: yf.Ticker(ticker).news, default=[]) or []
    texts = []
    for item in news:
        # yfinance news schema varies; try common keys
        title = item.get("title") or item.get("content", {}).get("title")
        if title:
            texts.append(title)
    return texts


def _score_texts(texts: list[str], sia) -> float:
    if not texts:
        return 0.0
    if sia is not None:
        return float(np.mean([sia.polarity_scores(t)["compound"] for t in texts]))
    # fallback bag-of-words
    vals = []
    for t in texts:
        toks = set(t.lower().split())
        vals.append(len(toks & _FALLBACK_POS) - len(toks & _FALLBACK_NEG))
    return float(np.mean(vals))


def sentiment_scores(tickers: list[str]) -> pd.Series:
    """Return z-scored news sentiment per ticker (0 = neutral/no data)."""
    sia = _analyzer()
    raw = {t: _score_texts(_headline_texts(t), sia) for t in tickers}
    s = pd.Series(raw)
    if s.abs().sum() == 0:
        return s  # no data anywhere -> all neutral
    return zscore(s)
