"""Edge-case tests for scoring helpers."""
import numpy as np
import pandas as pd
from utils import zscore, winsorize, rank_pct, safe


def test_zscore_normal():
    z = zscore(pd.Series([1.0, 2, 3, 4, 5]))
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=0) - 1) < 1e-9


def test_zscore_zero_variance():
    z = zscore(pd.Series([7.0, 7, 7, 7]))
    assert (z == 0).all()


def test_zscore_all_nan():
    z = zscore(pd.Series([np.nan, np.nan, np.nan]))
    assert (z == 0).all()


def test_zscore_single_element():
    z = zscore(pd.Series([42.0]))
    assert z.iloc[0] == 0.0


def test_zscore_with_some_nan():
    z = zscore(pd.Series([1.0, 2.0, np.nan, 4.0]))
    assert z.isna().sum() == 0            # NaNs filled with 0
    assert np.isfinite(z).all()


def test_winsorize_clips_outliers():
    s = pd.Series(list(range(100)) + [10_000])
    w = winsorize(s, 0.05)
    assert w.max() < 10_000


def test_rank_pct_range():
    r = rank_pct(pd.Series([10.0, 20, 30, 40]))
    assert r.min() >= 0 and r.max() <= 1
    assert r.idxmax() == 3


def test_rank_pct_all_nan_is_neutral():
    r = rank_pct(pd.Series([np.nan, np.nan]))
    assert (r == 0.5).all()


def test_safe_returns_default_on_error():
    assert safe(lambda: 1 / 0, default=-1) == -1
    assert safe(lambda: 5, default=-1) == 5
