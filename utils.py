"""utils.py — shared helpers for the screener (scoring, normalization)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _detect_month_end_alias() -> str:
    """pandas >=2.2 requires 'ME' for month-end resampling and rejects 'M';
    older pandas requires 'M'. Detect once so the code runs on both."""
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    try:
        pd.Series(0.0, index=idx).resample("ME").mean()
        return "ME"
    except (ValueError, KeyError):
        return "M"


# Month-end frequency alias valid on this pandas version.
MONTH_END = _detect_month_end_alias()


def zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score, robust to NaN and zero-variance."""
    s = s.astype(float)
    mu, sd = s.mean(), s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).fillna(0.0)


def winsorize(s: pd.Series, p: float = 0.05) -> pd.Series:
    """Clip extreme outliers to the p / 1-p quantiles."""
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def rank_pct(s: pd.Series) -> pd.Series:
    """Percentile rank in [0,1]; higher = better."""
    return s.rank(pct=True).fillna(0.5)


def safe(fn, default=np.nan):
    """Run fn(), return default on any exception (keeps pipeline alive)."""
    try:
        return fn()
    except Exception:
        return default
