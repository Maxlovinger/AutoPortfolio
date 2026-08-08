"""
fx/regime.py — HMM crash-regime overlay for the carry book (disciplined).

Carry's weakness is the crash tail, and carry crashes ARE a regime switch
(calm carry-earning state -> risk-off state where high-yielders get dumped).
This fits a 2-state Gaussian HMM on CRASH-RELEVANT features and scales the carry
book down by the FILTERED probability of the risk-off state.

Design rules (why this isn't just a slower vol-target):
  * FILTERED, not smoothed/Viterbi. The tradable signal P(stress_t | obs<=t)
    uses only past data (forward algorithm, frozen params) — reuses the
    exposure_models.py pattern. Viterbi (viterbi_states) is provided ONLY to
    label history for validation, never for the live signal.
  * CROSS-ASSET features, not the book's own returns: (1) fx_vol = cross-
    sectional average realized vol of the majors, (2) haven_spread = safe-haven
    basket (JPY, CHF) minus high-beta basket (AUD, NZD). These can flip risk-off
    BEFORE the carry book's own vol spikes — the thing plain vol-target can't do.

Anti-overfitting discipline:
  * Exactly 2 states, only 2 features, DIAGONAL covariance (few parameters).
  * Fit ONCE on the train window, freeze params, forward-filter the rest.
    Never refit with future data; fixed random_state.
  * Held-out TEST slice is the honest bar (run_holdout / __main__), same metrics
    as every other experiment. Graceful fallback to a causal vol-rule if
    hmmlearn is unavailable.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from fx.backtest_carry import (
    FREQ, resample_spot, run_carry_backtest, summarize, performance,
)
from fx.overlay import apply_overlay

HAVENS = ["JPY", "CHF"]
HIGH_BETA = ["AUD", "NZD"]


# --- features --------------------------------------------------------------
def crash_features(spot: pd.DataFrame, freq="M", vol_months=3,
                   tone_panel=None) -> pd.DataFrame:
    """Causal, cross-asset crash features on the trade grid. fx_vol rises in
    stress; haven_spread turns positive when havens outperform high-beta.

    Optional third feature `risk_tone` = NEGATIVE of broad cross-currency news
    tone, so it rises when news turns risk-off — a JUMP-anticipating input that
    realized vol can't provide (news deteriorates before vol spikes). fx_vol is
    kept as column 0 so the 'stress = higher fx_vol mean' identification holds."""
    px = resample_spot(spot, freq)
    ret = px.pct_change()
    ppy = FREQ[freq]["ppy"]
    k = max(2, round(vol_months / 12 * ppy))

    fx_vol = ret.rolling(k).std().mean(axis=1) * np.sqrt(ppy)
    havens = [c for c in HAVENS if c in ret.columns]
    hb = [c for c in HIGH_BETA if c in ret.columns]
    haven_spread = ret[havens].mean(axis=1) - ret[hb].mean(axis=1)

    feats = pd.DataFrame({"fx_vol": fx_vol, "haven_spread": haven_spread}).dropna()

    if tone_panel is not None and not tone_panel.empty:
        tp = tone_panel
        if getattr(tp.index, "tz", None) is not None:
            tp = tp.copy(); tp.index = tp.index.tz_localize(None)
        # broad risk-off intensity; ffill onto the grid, 0 where no tone yet
        risk_tone = (-tp.mean(axis=1)).reindex(feats.index).ffill().fillna(0.0)
        feats["risk_tone"] = risk_tone
    return feats


# --- causal filtered stress probability ------------------------------------
def _logsumexp(a):
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m))) if np.isfinite(m) else -np.inf


def causal_stress_prob(feats: pd.DataFrame, train_end) -> pd.Series:
    """Filtered P(stress | obs<=t): fit HMM on obs<=train_end, freeze, forward-
    filter all. Stress = state with the higher fx_vol mean. Falls back to a
    causal vol-percentile rule if hmmlearn is missing."""
    idx = feats.index
    X = feats.values
    train_mask = idx <= pd.Timestamp(train_end)
    try:
        from hmmlearn.hmm import GaussianHMM
        from scipy.stats import multivariate_normal

        Xtr = X[train_mask]
        model = GaussianHMM(n_components=2, covariance_type="diag",
                            n_iter=200, random_state=42).fit(Xtr)
        stress = int(np.argmax(model.means_[:, 0]))          # higher fx_vol
        covs = model.covars_                                  # (2, d, d) for diag
        logB = np.column_stack([
            multivariate_normal.logpdf(X, model.means_[k], covs[k],
                                       allow_singular=True)
            for k in range(2)])
        logA = np.log(model.transmat_ + 1e-300)
        logpi = np.log(model.startprob_ + 1e-300)
        n = len(X)
        la = np.full((n, 2), -np.inf)
        la[0] = logpi + logB[0]
        la[0] -= _logsumexp(la[0])
        for t in range(1, n):
            for j in range(2):
                la[t, j] = _logsumexp(la[t - 1] + logA[:, j]) + logB[t, j]
            la[t] -= _logsumexp(la[t])
        return pd.Series(np.exp(la[:, stress]), index=idx)
    except Exception:
        rv = feats["fx_vol"]
        thr = rv[train_mask].quantile(0.7)
        return (rv > thr).astype(float)


# --- Viterbi: VALIDATION / VISUALIZATION ONLY (non-causal) -----------------
def viterbi_states(feats: pd.DataFrame, train_end) -> pd.Series:
    """Most-likely state PATH via Viterbi over the WHOLE series (uses future
    data). For labeling/checking historical crash episodes ONLY — never feed
    this into a live or backtested position. Returns 1 = stress, 0 = calm."""
    from hmmlearn.hmm import GaussianHMM
    X = feats.values
    Xtr = X[feats.index <= pd.Timestamp(train_end)]
    model = GaussianHMM(n_components=2, covariance_type="diag",
                        n_iter=200, random_state=42).fit(Xtr)
    stress = int(np.argmax(model.means_[:, 0]))
    path = model.predict(X)                                  # <- Viterbi
    return pd.Series((path == stress).astype(int), index=feats.index)


# --- overlay ---------------------------------------------------------------
def regime_exposure(spot, freq="M", train_end="2019-12-31", floor=0.0,
                    stress_weight=1.0, vol_months=3, tone_panel=None) -> pd.Series:
    """Causal exposure multiplier = clip(1 - stress_weight * P(stress), floor, 1),
    lagged one period so the position uses only prior-period information.
    Pass tone_panel to add the news risk-off feature to the HMM."""
    feats = crash_features(spot, freq, vol_months, tone_panel=tone_panel)
    p = causal_stress_prob(feats, train_end)
    exp = (1 - stress_weight * p).clip(lower=floor, upper=1.0)
    return exp.shift(1).dropna()                             # no look-ahead


def run_regime_overlay(base_result: dict, spot, train_end="2019-12-31",
                       floor=0.0, stress_weight=1.0, cost_bps=5.0,
                       tone_panel=None) -> dict:
    """Apply the HMM crash overlay to a carry backtest result."""
    freq = base_result["freq"]
    r = base_result["net_ret"]
    exp = regime_exposure(spot, freq, train_end, floor, stress_weight,
                          tone_panel=tone_panel)
    net, k = apply_overlay(r, exp.reindex(r.index).ffill().fillna(1.0), cost_bps)
    net = net.dropna()
    return {
        "freq": freq,
        "net_ret": net,
        "gross_ret": base_result.get("gross_ret"),
        "turnover": base_result["turnover"],
        "exposure": k.reindex(net.index),
        "equity": (1 + net).cumprod(),
    }


# --- held-out evaluation ---------------------------------------------------
def run_holdout(spot, carry, train_end="2019-12-31", freq="M",
                stress_weight=1.0, cost_bps=5.0, tone_panel=None) -> dict:
    """Compare plain carry vs carry+regime (vol+haven) vs carry+regime+tone on
    the TEST slice (dates>train_end), the honest bar. HMM params fit only on
    <=train_end. The 3rd book (news risk-off feature) is added iff tone given."""
    base = run_carry_backtest(spot, carry, freq=freq, cost_bps=cost_bps)
    over = run_regime_overlay(base, spot, train_end=train_end,
                              stress_weight=stress_weight, cost_bps=cost_bps)
    books = [("carry", base), ("carry+regime", over)]
    if tone_panel is not None:
        over_t = run_regime_overlay(base, spot, train_end=train_end,
                                    stress_weight=stress_weight, cost_bps=cost_bps,
                                    tone_panel=tone_panel)
        books.append(("carry+regime+tone", over_t))

    ppy = FREQ[freq]["ppy"]
    test = pd.Timestamp(train_end)
    out = {}
    for name, res in books:
        full = summarize(res)
        tslice = performance(res["net_ret"][res["net_ret"].index > test], ppy)
        out[name] = {"full": full, "test": tslice}
    return out


if __name__ == "__main__":
    from fx.data import load_all
    from fx.sentiment import tone_panel_from_gdelt

    d = load_all(start="2010-01-01")
    spot, carry = d["spot"], d["carry"]

    # Path B: fetch broad country news tone (only ~10 GDELT queries) and add it
    # as a risk-off feature to the crash overlay. Needs a non-datacenter IP.
    print("Fetching country news tone (10 currencies) from GDELT...", flush=True)
    try:
        # GDELT DOC API coverage starts ~2017; 96m (8y) stays within it. The
        # printed per-economy span tells us how far back tone actually goes —
        # the HMM trains on <=train_end, so we need tone in that window.
        tone = tone_panel_from_gdelt(list(carry.columns) + ["USD"], timespan="96m")
        if tone.empty:
            tone = None
            print("  no tone (GDELT 429-blocked?). Showing vol+haven overlay only.")
        else:
            print(f"  tone for {tone.shape[1]} economies, "
                  f"{tone.index.min().date()}..{tone.index.max().date()}")
    except Exception as e:
        tone = None
        print(f"  tone fetch failed ({str(e)[:40]}); vol+haven overlay only.")

    res = run_holdout(spot, carry, train_end="2019-12-31", tone_panel=tone)

    print(f"\n{'':18}{'FULL':>26}{'TEST (>2019)':>26}")
    print(f"{'':18}{'sharpe':>9}{'maxDD':>9}{'skew':>8}{'sharpe':>9}{'maxDD':>9}{'skew':>8}")
    for name, m in res.items():
        f, t = m["full"], m["test"]
        print(f"{name:18}{f['sharpe']:>9.3f}{f['max_dd']:>9.3f}{f['skew']:>8.2f}"
              f"{t['sharpe']:>9.3f}{t['max_dd']:>9.3f}{t['skew']:>8.2f}")

    # Viterbi validation: does it flag the known crash episodes?
    feats = crash_features(spot)
    vit = viterbi_states(feats, "2019-12-31")
    flagged = vit[vit == 1].index
    print("\nViterbi-flagged stress months (validation only):")
    for yr in (2015, 2019, 2020, 2022):
        hits = [d.strftime("%Y-%m") for d in flagged if d.year == yr]
        print(f"  {yr}: {hits}")
