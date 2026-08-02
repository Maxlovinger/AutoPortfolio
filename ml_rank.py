"""
ml_rank.py — (E) Machine-learning learning-to-rank.

PRIMARY: LightGBM `LGBMRanker` with the LambdaMART (`lambdarank`) objective —
the proper learning-to-rank formulation. We group the panel BY DATE, assign a
per-date relevance label (quantile bucket of forward return), and let the model
learn to order stocks within each cross-section.

FALLBACK: sklearn GradientBoostingRegressor (regression on forward return) when
LightGBM is unavailable. Both return a z-scored score per ticker.

Panel: for every (date, stock) we compute price-derived features; the target is
the stock's forward N-day return.

WARNING — highest overfitting-risk component. Validate with the walk-forward
backtest; here it is blended with a small weight, never used alone.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from utils import zscore

FWD = 21          # predict ~1-month-ahead return
N_BUCKETS = 5     # relevance levels for the ranker
FEAT_COLS = ["mom_21", "mom_63", "mom_126", "mom_252",
             "vol_21", "vol_63", "revert", "dist_high"]

try:
    from lightgbm import LGBMRanker
    _HAVE_LGBM = True
except Exception:
    _HAVE_LGBM = False


def _features_for(prices_1: pd.Series) -> pd.DataFrame:
    """Price-derived features for a single ticker over time."""
    p = prices_1
    logret = np.log(p / p.shift(1))
    df = pd.DataFrame(index=p.index)
    df["mom_21"]  = p / p.shift(21) - 1
    df["mom_63"]  = p / p.shift(63) - 1
    df["mom_126"] = p / p.shift(126) - 1
    df["mom_252"] = p / p.shift(252) - 1
    df["vol_21"]  = logret.rolling(21).std()
    df["vol_63"]  = logret.rolling(63).std()
    df["revert"]  = -(p / p.rolling(21).mean() - 1)   # short-term reversal
    df["dist_high"] = p / p.rolling(252).max() - 1     # drawdown from 52w high
    return df


def build_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Stack per-ticker features + forward-return target into one panel."""
    frames = []
    for t in prices.columns:
        f = _features_for(prices[t])
        f["fwd_ret"] = prices[t].shift(-FWD) / prices[t] - 1
        f["ticker"] = t
        f["date"] = f.index
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["mom_252", "vol_63"])
    return panel


def _relevance_labels(g: pd.Series) -> pd.Series:
    """Quantile-bucket forward returns within a date into 0..N_BUCKETS-1."""
    if g.nunique() < 2:
        return pd.Series(0, index=g.index)
    try:
        return pd.qcut(g, min(N_BUCKETS, g.nunique()),
                       labels=False, duplicates="drop").astype(int)
    except Exception:
        return g.rank(method="first").astype(int) - 1


def _fit_lambdamart(train: pd.DataFrame):
    """LambdaMART ranker grouped by date."""
    train = train.sort_values("date")
    train = train.copy()
    train["rel"] = (train.groupby("date")["fwd_ret"]
                         .transform(_relevance_labels))
    group_sizes = train.groupby("date").size().values
    # need groups with >1 member for pairwise ranking
    if len(group_sizes) < 2 or (group_sizes > 1).sum() < 2:
        return None
    model = LGBMRanker(
        objective="lambdarank", n_estimators=300, num_leaves=15,
        learning_rate=0.03, subsample=0.8, min_child_samples=20,
        random_state=42, verbosity=-1,
    )
    model.fit(train[FEAT_COLS].values, train["rel"].values, group=group_sizes)
    return model


def _fit_gbr(train: pd.DataFrame):
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=42,
    )
    model.fit(train[FEAT_COLS].values, train["fwd_ret"].values)
    return model


def ml_rank_scores(prices: pd.DataFrame) -> pd.Series:
    """Train on history, return z-scored predicted rank score per ticker."""
    if prices.shape[1] == 0:
        return pd.Series(dtype=float)
    panel = build_panel(prices)
    train = panel.dropna(subset=["fwd_ret"])
    if len(train) < 300:
        return pd.Series(0.0, index=prices.columns)

    model = (_fit_lambdamart(train) if _HAVE_LGBM else None) or _fit_gbr(train)

    latest = (panel.dropna(subset=FEAT_COLS)
                    .sort_values("date")
                    .groupby("ticker").tail(1)
                    .set_index("ticker"))
    if latest.empty:
        return pd.Series(0.0, index=prices.columns)
    preds = pd.Series(model.predict(latest[FEAT_COLS].values),
                      index=latest.index)
    return zscore(preds).reindex(prices.columns).fillna(0.0)


def feature_importance(prices: pd.DataFrame) -> pd.Series:
    """
    Train the ranker and return its feature importances (for visualization).
    Returns all-zeros if there isn't enough data to train.
    """
    if prices.shape[1] == 0:
        return pd.Series(0.0, index=FEAT_COLS)
    panel = build_panel(prices)
    train = panel.dropna(subset=["fwd_ret"])
    if len(train) < 300:
        return pd.Series(0.0, index=FEAT_COLS)
    model = (_fit_lambdamart(train) if _HAVE_LGBM else None) or _fit_gbr(train)
    imp = np.asarray(getattr(model, "feature_importances_",
                             np.zeros(len(FEAT_COLS))), dtype=float)
    return pd.Series(imp, index=FEAT_COLS)
