"""
sentiment.py — (B) Alternative-data signal: news-headline sentiment (equities).

For each ticker we pull recent news headlines (via yfinance) and score their
tone, then z-score across the universe. Recent-tone shifts often lead price.

Scoring now delegates to news_sentiment (shared with the FX sleeve): FinBERT by
default — finance-tuned, much stronger on financial text than a general
lexicon — with automatic VADER / bag-of-words fallback if the model isn't
available. Pass prefer="vader" to force the lightweight path.

Graceful degradation: if the news feed or scorer is unavailable, every ticker
gets a neutral 0 and the pipeline continues.

NOTE (point-in-time): yfinance news is CURRENT headlines only (no history), so
this signal is live/forward-safe but NOT backtestable. For a backtestable news
signal use the GDELT tone backbone (news_gdelt.entity_tone).
"""
from __future__ import annotations
import pandas as pd
import yfinance as yf

from utils import zscore, safe
from news_sentiment import aggregate


def _headline_texts(ticker: str) -> list[str]:
    news = safe(lambda: yf.Ticker(ticker).news, default=[]) or []
    texts = []
    for item in news:
        # yfinance news schema varies; try common keys
        title = item.get("title") or item.get("content", {}).get("title")
        if title:
            texts.append(title)
    return texts


def sentiment_scores(tickers: list[str], prefer="finbert") -> pd.Series:
    """Return z-scored news sentiment per ticker (0 = neutral/no data)."""
    raw = {t: aggregate(_headline_texts(t), prefer=prefer) for t in tickers}
    s = pd.Series(raw)
    if s.abs().sum() == 0:
        return s  # no data anywhere -> all neutral
    return zscore(s)
