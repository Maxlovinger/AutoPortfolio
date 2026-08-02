"""
allocators.py — risk-based portfolio weighting schemes (no return forecasts).

Our disciplined backtest showed signal-picking (reversal) doesn't beat a plain
equal-weight benchmark once survivorship bias + realistic costs are honest. That
points at a different lever: HOW you weight a diversified basket. These allocators
use only the covariance of past returns — no fragile expected-return estimates,
no fundamentals — so they're testable on free data and inherently low-turnover:

  equal_weight        1/N — the champion to beat.
  inverse_vol         w ∝ 1/vol; cheap "risk parity lite".
  min_variance        lowest portfolio variance (needs a stable covariance).
  erc                 Equal Risk Contribution — each name adds equal risk.
  max_diversification Choueifaty: maximize (w·σ)/σ_p, the diversification ratio.
  hrp                 Hierarchical Risk Parity (López de Prado): cluster the
                      correlation tree, then split risk down it. Robust to the
                      noisy, near-singular covariances you get in big universes —
                      no matrix inversion required.

All take a trailing-returns DataFrame (dates × tickers) and return weights that
sum to 1, indexed by ticker.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

TRADING_DAYS = 252


def _clean(rets: pd.DataFrame) -> pd.DataFrame:
    return rets.dropna(axis=1, how="all").fillna(0.0)


def _cov(rets: pd.DataFrame, shrink: bool = False) -> np.ndarray:
    """Annualized covariance. With shrink=True, Ledoit-Wolf shrinkage toward a
    scaled identity — the standard cure for noisy, near-singular sample
    covariance in wide universes (stabilizes min-var / ERC / max-div / HRP)."""
    if shrink:
        from sklearn.covariance import LedoitWolf
        X = rets.fillna(0.0).values
        return LedoitWolf().fit(X).covariance_ * TRADING_DAYS
    return rets.cov().values * TRADING_DAYS


def _vol(rets: pd.DataFrame) -> np.ndarray:
    return rets.std(ddof=0).values * np.sqrt(TRADING_DAYS)


def equal_weight(rets: pd.DataFrame) -> pd.Series:
    n = rets.shape[1]
    return pd.Series(np.repeat(1.0 / n, n), index=rets.columns)


def inverse_vol(rets: pd.DataFrame) -> pd.Series:
    v = _vol(rets)
    iv = np.where(v > 0, 1.0 / v, 0.0)
    s = iv.sum()
    w = iv / s if s > 0 else np.repeat(1.0 / len(iv), len(iv))
    return pd.Series(w, index=rets.columns)


def min_variance(rets: pd.DataFrame, cap: float = 1.0,
                 shrink: bool = False) -> pd.Series:
    Sig = _cov(rets, shrink=shrink)
    n = len(Sig)
    cap_eff = max(cap, 1.0 / n)
    res = minimize(lambda w: w @ Sig @ w, np.repeat(1.0 / n, n), method="SLSQP",
                   bounds=[(0, cap_eff)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return pd.Series(res.x, index=rets.columns)


def erc(rets: pd.DataFrame, shrink: bool = False) -> pd.Series:
    """Equal Risk Contribution (a.k.a. risk parity).

    Uses the standard convex reformulation (Maillard/Spinu): minimize
    ½·wᵀΣw − (1/n)·Σ log wᵢ over w > 0, then normalize. Its unique optimum has
    every asset contributing equal risk, and it converges far more reliably than
    directly penalizing risk-contribution dispersion."""
    Sig = _cov(rets, shrink=shrink)
    n = len(Sig)

    def obj(w):
        return 0.5 * (w @ Sig @ w) - (1.0 / n) * np.sum(np.log(w))

    res = minimize(obj, np.repeat(1.0 / n, n), method="SLSQP",
                   bounds=[(1e-8, None)] * n)
    w = np.maximum(res.x, 0.0)
    return pd.Series(w / w.sum(), index=rets.columns)


def max_diversification(rets: pd.DataFrame, shrink: bool = False) -> pd.Series:
    """Choueifaty maximum-diversification portfolio."""
    Sig = _cov(rets, shrink=shrink)
    vol = np.sqrt(np.diag(Sig))
    n = len(Sig)

    def neg_dr(w):
        pv = np.sqrt(w @ Sig @ w)
        return -(w @ vol) / pv if pv > 0 else 0.0

    res = minimize(neg_dr, np.repeat(1.0 / n, n), method="SLSQP",
                   bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return pd.Series(res.x, index=rets.columns)


# ---- Hierarchical Risk Parity (López de Prado 2016) ----
def _quasi_diag(link: np.ndarray) -> list[int]:
    link = link.astype(int)
    n = link[-1, 3]                        # total original items
    order = pd.Series([link[-1, 0], link[-1, 1]])
    while order.max() >= n:
        order.index = range(0, order.shape[0] * 2, 2)   # make space
        clusters = order[order >= n]
        i = clusters.index
        j = clusters.values - n
        order[i] = link[j, 0]              # left child
        order = pd.concat([order, pd.Series(link[j, 1], index=i + 1)])
        order = order.sort_index()
        order.index = range(order.shape[0])
    return order.tolist()


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    sub = cov.loc[items, items].values
    iv = 1.0 / np.diag(sub)
    w = iv / iv.sum()
    return float(w @ sub @ w)


def hrp(rets: pd.DataFrame, shrink: bool = False) -> pd.Series:
    rets = _clean(rets)
    cols = rets.columns
    if len(cols) == 1:
        return pd.Series([1.0], index=cols)
    corr = rets.corr().fillna(0.0).values
    covdf = pd.DataFrame(_cov(rets, shrink=shrink), index=cols, columns=cols)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    sort_ix = [cols[i] for i in _quasi_diag(link)]

    w = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]
    while clusters:
        clusters = [c[j:k] for c in clusters
                    for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
                    if len(c) > 1]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            vl, vr = _cluster_var(covdf, left), _cluster_var(covdf, right)
            alpha = 1.0 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= (1.0 - alpha)
    return (w / w.sum()).reindex(cols).fillna(0.0)


ALLOCATORS = {
    "equal": equal_weight,
    "inverse_vol": inverse_vol,
    "min_var": min_variance,
    "erc": erc,
    "max_div": max_diversification,
    "hrp": hrp,
}


def allocate(rets: pd.DataFrame, method: str = "equal", cap: float = 1.0,
             shrink: bool = False) -> pd.Series:
    rets = _clean(rets)
    if rets.shape[1] == 0:
        return pd.Series(dtype=float)
    if rets.shape[1] == 1:
        return pd.Series([1.0], index=rets.columns)
    if method in ("equal", "inverse_vol"):
        return ALLOCATORS[method](rets)
    if method == "min_var":
        return min_variance(rets, cap=cap, shrink=shrink)
    return ALLOCATORS[method](rets, shrink=shrink)
