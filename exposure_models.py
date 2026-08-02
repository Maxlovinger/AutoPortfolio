"""
exposure_models.py — "how much invested vs. how much in cash" decision models.

The volatility-targeting overlay answers this by scaling exposure to hit a risk
target using only volatility. This module implements and compares the richer,
return-aware alternatives head-to-head, all strictly NON-ANTICIPATING (exposure
for day t uses only information through t-1):

  vol-target (realized / EWMA-0.94 / EWMA-MLE) : risk-based (the baseline family).
  trend / time-series momentum                 : invested when the book's own
        equity is above its 200-day average, in cash (or a floor) below it.
  regime (HMM)                                 : cut exposure when a causal Hidden
        Markov filter says we're in the high-stress regime.
  CPPI                                          : Constant Proportion Portfolio
        Insurance — exposure ∝ cushion above a ratcheting floor; guarantees a
        soft max-drawdown near (1 - floor).

Each model produces an EXPOSURE series in [0, 1] that multiplies the strategy's
raw daily returns (1 = fully invested, 0 = all cash). CPPI is path-dependent so
it's simulated day by day.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# EWMA lambda via maximum likelihood (Question A)
# ----------------------------------------------------------------------
def _ewma_var(returns: np.ndarray, lam: float) -> np.ndarray:
    """EWMA variance forecast; var[t] uses returns through t-1 (non-anticipating)."""
    n = len(returns)
    var = np.empty(n)
    seed = np.nanvar(returns[:21]) if n >= 21 else np.nanvar(returns)
    prev = seed if np.isfinite(seed) and seed > 0 else float(np.var(returns) or 1e-6)
    for t in range(n):
        var[t] = prev
        prev = lam * prev + (1 - lam) * returns[t] ** 2
    return var


def fit_ewma_lambda(returns: pd.Series, bounds=(0.80, 0.995)) -> float:
    """Maximum-likelihood λ (Gaussian NLL of returns given EWMA variance).
    EWMA is IGARCH(1,1) with α=1-λ, β=λ; this is the principled way to pick λ —
    NOT linear regression (λ enters recursively and r² is a noisy variance proxy)."""
    r = returns.dropna().values

    def nll(lam):
        v = _ewma_var(r, lam)
        v = np.maximum(v, 1e-12)
        return 0.5 * np.sum(np.log(v) + r ** 2 / v)

    res = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(res.x)


def ewma_vol_series(returns: pd.Series, lam: float) -> pd.Series:
    v = _ewma_var(returns.fillna(0.0).values, lam)
    return pd.Series(np.sqrt(v) * np.sqrt(TRADING_DAYS), index=returns.index)


def realized_vol_series(returns: pd.Series, lookback=21) -> pd.Series:
    return (returns.rolling(lookback).std() * np.sqrt(TRADING_DAYS)).shift(1)


# ----------------------------------------------------------------------
# Exposure models (each returns an exposure Series in [0, max_lev])
# ----------------------------------------------------------------------
def exposure_voltarget(vol_series: pd.Series, target=0.15, max_lev=1.0) -> pd.Series:
    return (target / vol_series).clip(upper=max_lev).fillna(max_lev)


def exposure_trend(returns: pd.Series, window=200, floor=0.0, max_lev=1.0) -> pd.Series:
    """Time-series momentum on the strategy's own equity curve (Faber-style).
    Invested at max_lev when equity > its `window`-day average, else `floor`."""
    equity = (1 + returns.fillna(0.0)).cumprod()
    sma = equity.rolling(window).mean()
    signal = (equity > sma).astype(float)
    exp = floor + (max_lev - floor) * signal
    return exp.shift(1).fillna(max_lev)          # decide on yesterday's close


def causal_stress_prob(returns: pd.Series, train_end: str) -> pd.Series:
    """Causal (no look-ahead) P(stress regime) via a 2-state Gaussian HMM.

    Fit the HMM ONCE on the training window, then forward-FILTER over the full
    series using the frozen parameters — filtered α_t uses only obs 1..t, so no
    future leaks in (unlike smoothed probabilities). Falls back to a causal
    vol-rule if hmmlearn is unavailable."""
    r = returns.fillna(0.0)
    vol = r.rolling(21).std().bfill()
    X = np.column_stack([r.values * 100.0, vol.values * 100.0])
    try:
        from hmmlearn.hmm import GaussianHMM
        Xtr = X[returns.index <= pd.Timestamp(train_end)]
        model = GaussianHMM(n_components=2, covariance_type="full",
                            n_iter=200, random_state=42).fit(Xtr)
        # stress = higher-variance state (mean of vol feature)
        stress = int(np.argmax(model.means_[:, 1]))
        # manual forward filter with frozen params -> filtered P(state|obs<=t)
        from scipy.stats import multivariate_normal
        logB = np.column_stack([
            multivariate_normal.logpdf(X, model.means_[k], model.covars_[k],
                                       allow_singular=True)
            for k in range(2)])
        logA = np.log(model.transmat_ + 1e-300)
        logpi = np.log(model.startprob_ + 1e-300)
        n = len(X)
        logalpha = np.full((n, 2), -np.inf)
        logalpha[0] = logpi + logB[0]
        logalpha[0] -= _logsumexp(logalpha[0])
        for t in range(1, n):
            for j in range(2):
                logalpha[t, j] = _logsumexp(logalpha[t - 1] + logA[:, j]) + logB[t, j]
            logalpha[t] -= _logsumexp(logalpha[t])
        p = np.exp(logalpha[:, stress])
        return pd.Series(p, index=returns.index)
    except Exception:
        rv = r.rolling(21).std()
        med = rv[returns.index <= pd.Timestamp(train_end)].median()
        return (rv > med * 1.3).astype(float).reindex(returns.index).fillna(0.0)


def _logsumexp(a):
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m))) if np.isfinite(m) else -np.inf


def exposure_regime(returns: pd.Series, train_end: str, floor=0.0,
                    max_lev=1.0) -> pd.Series:
    """Exposure = max_lev when calm, scaled down toward `floor` by stress prob."""
    p = causal_stress_prob(returns, train_end)
    exp = max_lev - (max_lev - floor) * p
    return exp.shift(1).fillna(max_lev)


def cppi_returns(returns: pd.Series, m=3.0, floor_frac=0.80, max_lev=1.0):
    """Constant Proportion Portfolio Insurance with a ratcheting floor (fraction
    of the running peak). exposure = clip(m·cushion/value, 0, max_lev). Simulated
    day by day (path-dependent). Returns (overlaid_returns, exposure)."""
    r = returns.fillna(0.0).values
    n = len(r)
    exp = np.empty(n)
    out = np.empty(n)
    V = 1.0
    peak = 1.0
    for t in range(n):
        floor = floor_frac * peak
        cushion = max(V - floor, 0.0)
        e = np.clip(m * cushion / V, 0.0, max_lev) if V > 0 else 0.0
        exp[t] = e
        out[t] = e * r[t]
        V *= (1 + out[t])
        peak = max(peak, V)
    idx = returns.index
    return pd.Series(out, index=idx), pd.Series(exp, index=idx)


def exposure_vix_breaker(returns: pd.Series, vix: pd.Series, threshold=0.35,
                         floor=0.3, ramp=0.0, max_lev=1.0) -> pd.Series:
    """Tail-risk circuit breaker: ignore the VIX entirely until it exceeds
    `threshold` (a fixed level, e.g. 0.35 = VIX 35), then cut exposure to `floor`.
    With ramp>0 the cut is graduated over [threshold, threshold+ramp] instead of
    a hard switch. Uses YESTERDAY's VIX (shift 1) — non-anticipating."""
    v = vix.reindex(returns.index).ffill().shift(1)
    if ramp <= 0:
        exp = pd.Series(max_lev, index=returns.index)
        exp[v > threshold] = floor
    else:
        over = ((v - threshold) / ramp).clip(0, 1)
        exp = max_lev - (max_lev - floor) * over
    return exp.fillna(max_lev)


def exposure_vix_percentile(returns: pd.Series, vix: pd.Series, q=0.90,
                            floor=0.3, min_obs=252, max_lev=1.0) -> pd.Series:
    """Circuit breaker keyed to the VIX's OWN history: cut when the VIX exceeds
    its trailing q-quantile (expanding, past-only — no look-ahead in the
    threshold). Adapts to whatever 'extreme' has meant historically."""
    v = vix.reindex(returns.index).ffill()
    thr = v.expanding(min_periods=min_obs).quantile(q)
    trig = (v > thr).shift(1).fillna(False)
    exp = pd.Series(max_lev, index=returns.index)
    exp[trig] = floor
    return exp


def banded_exposure(target_exposure: pd.Series, band=0.10, check_every=1) -> pd.Series:
    """Turn a continuously-varying target exposure into a tradeable one that only
    moves when it drifts beyond a no-trade band, and only on check days.

    band        : only re-trade exposure when |target - held| > band (e.g. 0.10).
    check_every : days between checks (1 = daily, 5 = weekly, 63 = quarterly).

    Reduces exposure turnover (and cost) while preserving most of the risk
    control. `target_exposure` must already be non-anticipating."""
    t = target_exposure.values
    n = len(t)
    held = np.empty(n)
    cur = t[0] if (n and np.isfinite(t[0])) else 1.0
    for i in range(n):
        if i % check_every == 0 and np.isfinite(t[i]) and abs(t[i] - cur) > band:
            cur = t[i]
        held[i] = cur
    return pd.Series(held, index=target_exposure.index)


def apply_exposure(returns: pd.Series, exposure: pd.Series):
    e = exposure.reindex(returns.index).fillna(1.0)
    return returns * e, e
