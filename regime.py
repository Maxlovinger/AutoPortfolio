"""
regime.py — (D) Regime-switching model.

PRIMARY: a 2-state Gaussian Hidden Markov Model (hmmlearn) fit to broad-market
(default SPY) daily returns + rolling volatility. The states self-identify as a
CALM regime and a STRESS regime.

FALLBACKS (in order): statsmodels Markov-switching variance model, then a plain
realized-volatility rule. The system always returns a regime.

CRUCIAL correctness detail: we identify which state is "stress" DATA-DRIVEN — by
measuring the empirical return variance of the observations assigned to each
state — rather than relying on any library's internal parameter ordering.

The regime then ADAPTS the screener's factor weights:
  CALM   -> lean into momentum + value x momentum
  STRESS -> lean into quality + low-vol + the network diversifier
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

from utils import safe


def _market_returns(market_ticker: str, start: str):
    px = safe(lambda: yf.download(market_ticker, start=start,
                                  auto_adjust=True, progress=False)["Close"])
    if px is None or len(px) < 250:
        return None
    ret = np.log(px / px.shift(1)).dropna()
    return ret.squeeze()


def _identify_stress(ret: pd.Series, states: np.ndarray) -> int:
    """Return the state index whose assigned returns have the HIGHER variance."""
    variances = {}
    for s in np.unique(states):
        variances[s] = float(np.var(ret.values[states == s]))
    return int(max(variances, key=variances.get))


def _hmm_regime(ret: pd.Series) -> dict | None:
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception:
        return None
    r = ret.values.reshape(-1, 1) * 100.0
    vol = pd.Series(ret.values).rolling(21).std().bfill().values.reshape(-1, 1) * 100.0
    X = np.hstack([r, vol])
    try:
        model = GaussianHMM(n_components=2, covariance_type="full",
                            n_iter=200, random_state=42)
        model.fit(X)
        states = model.predict(X)
        stress = _identify_stress(ret, states)
        post = model.predict_proba(X)[-1]
        p_stress = float(post[stress])
        return {"regime": "stress" if p_stress > 0.5 else "calm",
                "p_stress": p_stress, "p_calm": 1 - p_stress, "method": "hmm"}
    except Exception:
        return None


def _markov_regime(ret: pd.Series) -> dict | None:
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except Exception:
        return None
    try:
        r = ret.values * 100.0
        res = MarkovRegression(r, k_regimes=2, trend="c",
                               switching_variance=True).fit(disp=False)
        smp = res.smoothed_marginal_probabilities
        states = np.asarray(smp).argmax(axis=1)
        stress = _identify_stress(ret, states)
        p_stress = float(np.asarray(smp)[:, stress][-1])
        return {"regime": "stress" if p_stress > 0.5 else "calm",
                "p_stress": p_stress, "p_calm": 1 - p_stress, "method": "markov"}
    except Exception:
        return None


def _vol_fallback(ret: pd.Series) -> dict:
    rv = ret.rolling(21).std().dropna()
    if len(rv) == 0:
        return {"regime": "calm", "p_stress": 0.0, "p_calm": 1.0, "method": "none"}
    cur, med = float(rv.iloc[-1]), float(rv.median())
    p_stress = 1.0 if cur > med * 1.3 else 0.0
    return {"regime": "stress" if p_stress else "calm",
            "p_stress": p_stress, "p_calm": 1 - p_stress, "method": "vol_fallback"}


def detect_regime(market_ticker: str = "SPY", start: str = "2015-01-01",
                  returns: pd.Series | None = None) -> dict:
    """
    Detect the current market regime. `returns` may be supplied directly
    (bypasses the network download) — used by tests and backtests.
    """
    ret = returns if returns is not None else _market_returns(market_ticker, start)
    if ret is None or len(ret) < 250:
        return {"regime": "calm", "p_stress": 0.0, "p_calm": 1.0, "method": "none"}
    return _hmm_regime(ret) or _markov_regime(ret) or _vol_fallback(ret)


def regime_probabilities(returns: pd.Series) -> pd.Series:
    """
    Per-timestep probability of being in the STRESS regime, aligned to
    `returns.index`. Used to VISUALIZE the regime over time. Same engine
    priority as detect_regime (HMM -> Markov -> vol rule).
    """
    ret = returns
    if ret is None or len(ret) < 250:
        return pd.Series(dtype=float)

    # HMM timeline
    try:
        from hmmlearn.hmm import GaussianHMM
        r = ret.values.reshape(-1, 1) * 100.0
        vol = pd.Series(ret.values).rolling(21).std().bfill().values.reshape(-1, 1) * 100.0
        X = np.hstack([r, vol])
        model = GaussianHMM(n_components=2, covariance_type="full",
                            n_iter=200, random_state=42).fit(X)
        states = model.predict(X)
        stress = _identify_stress(ret, states)
        return pd.Series(model.predict_proba(X)[:, stress], index=ret.index)
    except Exception:
        pass

    # Markov timeline
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        res = MarkovRegression(ret.values * 100.0, k_regimes=2, trend="c",
                               switching_variance=True).fit(disp=False)
        smp = np.asarray(res.smoothed_marginal_probabilities)
        stress = _identify_stress(ret, smp.argmax(axis=1))
        return pd.Series(smp[:, stress], index=ret.index)
    except Exception:
        pass

    # vol-rule fallback (0/1 indicator)
    rv = ret.rolling(21).std()
    p = (rv > rv.median() * 1.3).astype(float)
    return pd.Series(p.values, index=ret.index).fillna(0.0)


def regime_factor_weights(regime: dict) -> dict:
    """Map regime probability to factor weights (linear calm<->stress blend)."""
    p = float(np.clip(regime.get("p_stress", 0.0), 0.0, 1.0))
    calm =   {"value": 0.8, "quality": 0.8, "momentum": 1.3, "value_x_mom": 0.8,
              "sentiment": 0.5, "network": 0.3}
    stress = {"value": 1.0, "quality": 1.5, "momentum": 0.4, "value_x_mom": 0.3,
              "sentiment": 0.3, "network": 0.8}
    return {k: (1 - p) * calm[k] + p * stress[k] for k in calm}
