"""
news_sentiment.py — shared text-sentiment scorer for BOTH the equity and FX
sleeves. Turns raw headline/statement text into a compound score in [-1, +1].

Primary: FinBERT (ProsusAI/finbert), a finance-tuned transformer — far better
than a general lexicon on financial text ("beats guidance" vs "fraud probe").
    compound = P(positive) - P(negative)      (neutral pulls toward 0)

Fallback: VADER (the project's existing analyzer), then a tiny bag-of-words, so
this never hard-fails offline — it just degrades to a weaker signal.

Use this ONLY on raw text you fetch (yfinance headlines, central-bank
statements, NewsAPI). GDELT already ships a pre-computed tone, so that path
skips the scorer (see news_gdelt.py).
"""
from __future__ import annotations
import numpy as np

_FALLBACK_POS = {"beat", "surge", "record", "upgrade", "growth", "strong",
                 "buy", "gain", "hawkish", "rally", "raises"}
_FALLBACK_NEG = {"miss", "cut", "downgrade", "lawsuit", "probe", "weak", "sell",
                 "loss", "fraud", "dovish", "plunge", "slump", "recession"}

# module-level singletons so we load the ~440MB model at most once per process
_FINBERT = None
_VADER = None


# --- FinBERT ---------------------------------------------------------------
class _FinBert:
    """Lazy FinBERT wrapper. compound = P(pos) - P(neg), batched inference."""

    def __init__(self):
        import torch
        from transformers import (AutoTokenizer,
                                   AutoModelForSequenceClassification)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.mdl = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert")
        self.mdl.eval()
        # map label name -> index, robust to id2label ordering
        self.idx = {v.lower(): k for k, v in self.mdl.config.id2label.items()}

    def score(self, texts: list[str], batch_size=32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=128)
            with self.torch.no_grad():
                probs = self.torch.softmax(self.mdl(**enc).logits, dim=-1)
            p = probs[:, self.idx["positive"]] - probs[:, self.idx["negative"]]
            out.extend(p.tolist())
        return np.asarray(out, dtype=float)


def _get_finbert():
    global _FINBERT
    if _FINBERT is None:
        _FINBERT = _FinBert()          # raises if transformers/torch/model absent
    return _FINBERT


# --- VADER / bag-of-words fallbacks ----------------------------------------
def _get_vader():
    global _VADER
    if _VADER is not None:
        return _VADER
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER = SentimentIntensityAnalyzer()
    except Exception:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            _VADER = SentimentIntensityAnalyzer()
        except Exception:
            _VADER = False
    return _VADER


def _bow_score(text: str) -> float:
    toks = set(text.lower().split())
    return float(len(toks & _FALLBACK_POS) - len(toks & _FALLBACK_NEG))


# --- Public API ------------------------------------------------------------
def score_texts(texts, prefer="finbert") -> np.ndarray:
    """
    Compound sentiment per text in [-1, 1]. `prefer` in {"finbert","vader"}.
    Falls back finbert -> vader -> bag-of-words, so it always returns something.
    """
    texts = [t for t in (texts or []) if t]
    if not texts:
        return np.array([], dtype=float)

    if prefer == "finbert":
        try:
            return _get_finbert().score(texts)
        except Exception:
            pass  # fall through to VADER

    vader = _get_vader()
    if vader:
        return np.array([vader.polarity_scores(t)["compound"] for t in texts])
    return np.array([_bow_score(t) for t in texts])


def aggregate(texts, prefer="finbert") -> float:
    """Mean compound over a batch of texts (0.0 if none). The per-entity signal."""
    s = score_texts(texts, prefer=prefer)
    return float(s.mean()) if len(s) else 0.0


def available_backend(prefer="finbert") -> str:
    """Which scorer would actually be used — handy for logging/tests."""
    if prefer == "finbert":
        try:
            _get_finbert()
            return "finbert"
        except Exception:
            pass
    return "vader" if _get_vader() else "bagofwords"
