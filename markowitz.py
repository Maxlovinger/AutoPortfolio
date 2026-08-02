"""
markowitz.py — Mean-Variance Portfolio optimization (the core of MPT).

This is written from scratch (numpy + scipy) on purpose, so the mechanics of
the efficient frontier are transparent rather than hidden in a library.

The three things Markowitz needs:
    mu  : expected annual return of each asset        (vector, length N)
    Sig : annualized covariance matrix of returns     (N x N)
    w   : the WEIGHTS we solve for                     (vector, length N, sums to 1)

Portfolio return     = w · mu
Portfolio variance   = w · Sig · w
Portfolio volatility = sqrt(variance)
Sharpe ratio         = (return - risk_free) / volatility

Markowitz does NOT pick assets. It finds the best WEIGHTS among the assets you
give it. Asset/universe selection is a separate, upstream step.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize


# ----------------------------------------------------------------------
# Basic portfolio math
# ----------------------------------------------------------------------
def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)


def portfolio_vol(w: np.ndarray, Sig: np.ndarray) -> float:
    return float(np.sqrt(w @ Sig @ w))


def sharpe_ratio(w, mu, Sig, rf=0.0) -> float:
    return (portfolio_return(w, mu) - rf) / portfolio_vol(w, Sig)


# ----------------------------------------------------------------------
# The three canonical portfolios
# ----------------------------------------------------------------------
def _base_constraints(n):
    # weights must sum to 1 (fully invested)
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]


def min_variance(Sig, allow_short=False):
    """The single lowest-risk portfolio (left tip of the frontier)."""
    n = len(Sig)
    bounds = None if allow_short else [(0.0, 1.0)] * n
    res = minimize(
        lambda w: w @ Sig @ w,
        x0=np.repeat(1 / n, n),
        method="SLSQP",
        bounds=bounds,
        constraints=_base_constraints(n),
    )
    return res.x


def max_sharpe(mu, Sig, rf=0.0, allow_short=False):
    """The tangency portfolio — best risk-adjusted return."""
    n = len(mu)
    bounds = None if allow_short else [(0.0, 1.0)] * n
    # minimize the NEGATIVE Sharpe ratio
    res = minimize(
        lambda w: -sharpe_ratio(w, mu, Sig, rf),
        x0=np.repeat(1 / n, n),
        method="SLSQP",
        bounds=bounds,
        constraints=_base_constraints(n),
    )
    return res.x


def efficient_return(mu, Sig, target, allow_short=False):
    """Lowest-risk portfolio that achieves a given target return."""
    n = len(mu)
    bounds = None if allow_short else [(0.0, 1.0)] * n
    cons = _base_constraints(n) + [
        {"type": "eq", "fun": lambda w: w @ mu - target}
    ]
    res = minimize(
        lambda w: w @ Sig @ w,
        x0=np.repeat(1 / n, n),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
    )
    return res.x


# ----------------------------------------------------------------------
# The efficient frontier
# ----------------------------------------------------------------------
def efficient_frontier(mu, Sig, n_points=50, allow_short=False):
    """
    Trace the EFFICIENT frontier: for a grid of target returns from the
    minimum-variance portfolio's return up to the max asset return, find the
    min-risk portfolio at each. Returns (vols, rets, weights_list).

    Note: we start at the min-variance return (not mu.min()) so we trace only
    the *efficient* upper branch — where higher return costs higher risk.
    Starting at mu.min() would include the inefficient lower branch.
    """
    w_mv = min_variance(Sig, allow_short)
    lo = portfolio_return(w_mv, mu)
    hi = mu.max()
    if hi <= lo:                      # degenerate (near-identical returns)
        lo, hi = min(lo, mu.min()), max(hi, mu.max())
    targets = np.linspace(lo, hi, n_points)
    vols, rets, weights = [], [], []
    for t in targets:
        w = efficient_return(mu, Sig, t, allow_short)
        vols.append(portfolio_vol(w, Sig))
        rets.append(portfolio_return(w, mu))
        weights.append(w)
    return np.array(vols), np.array(rets), weights
