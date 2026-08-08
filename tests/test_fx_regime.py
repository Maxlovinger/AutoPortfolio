"""
Offline tests for fx/regime.py — the HMM crash overlay. Synthetic spot with an
engineered calm->stress->calm structure so feature construction, the CAUSAL
filtered stress prob (no look-ahead), Viterbi labeling, and the overlay's
de-risking all check exactly.
"""
import numpy as np
import pandas as pd
import pytest

import fx.regime as reg
from fx.backtest_carry import run_carry_backtest
from tests.test_fx_carry import make_carry_world


def make_regime_spot(seed=0):
    """G10-ish spot with a clear high-vol, haven-rallying STRESS block in the
    middle (months ~40-55), calm otherwise."""
    rng = np.random.default_rng(seed)
    n = 1600
    idx = pd.bdate_range("2016-01-01", periods=n)
    ccys = ["EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK"]
    # stress window in business-day space (~months 40-55)
    stress = np.zeros(n, dtype=bool)
    stress[820:1130] = True
    spot = {}
    for c in ccys:
        base_vol = 0.004
        vol = np.where(stress, base_vol * 4, base_vol)      # vol spikes in stress
        drift = np.zeros(n)
        if c in ("JPY", "CHF"):        # havens rally in stress
            drift = np.where(stress, 0.0010, 0.0)
        if c in ("AUD", "NZD"):        # high-beta sells off in stress
            drift = np.where(stress, -0.0010, 0.0)
        shocks = rng.normal(drift, vol, n)
        spot[c] = np.exp(np.cumsum(shocks))
    return pd.DataFrame(spot, index=idx), idx, stress


def make_regime_tone(spot, idx, stress, tz=False):
    """Monthly country-tone panel: tone turns NEGATIVE during the stress window
    (so risk_tone = -mean turns positive there). Optionally tz-aware (like GDELT)."""
    m_idx = pd.date_range(spot.index[0], spot.index[-1],
                          freq="M", tz="UTC" if tz else None)
    ccys = list(spot.columns)
    # per business-day stress -> monthly stress fraction, mapped to tone
    stress_ser = pd.Series(stress.astype(float), index=idx).resample("M").mean()
    stress_m = stress_ser.reindex(m_idx if not tz else m_idx.tz_localize(None)).fillna(0.0)
    data = {c: (0.3 - 1.2 * stress_m.values) for c in ccys}   # +0.3 calm, -0.9 stress
    return pd.DataFrame(data, index=m_idx)


# --- features --------------------------------------------------------------
def test_crash_features_rise_in_stress():
    spot, idx, stress = make_regime_spot()
    feats = reg.crash_features(spot, freq="M")
    # map monthly feature index back to whether it's in the stress window
    m_end = feats.index
    # find fx_vol in stress months vs calm months
    stress_months = feats["fx_vol"][(m_end >= idx[850]) & (m_end <= idx[1100])]
    calm_months = feats["fx_vol"][(m_end < idx[800]) | (m_end > idx[1200])]
    assert stress_months.mean() > calm_months.mean() * 1.5


def test_haven_spread_positive_in_stress():
    spot, idx, stress = make_regime_spot()
    feats = reg.crash_features(spot, freq="M")
    stress_months = feats["haven_spread"][(feats.index >= idx[850]) &
                                          (feats.index <= idx[1100])]
    assert stress_months.mean() > 0          # havens outperform high-beta


# --- causal stress prob ----------------------------------------------------
def test_stress_prob_higher_in_stress_window():
    spot, idx, stress = make_regime_spot()
    feats = reg.crash_features(spot, freq="M")
    p = reg.causal_stress_prob(feats, train_end=str(idx[-1].date()))
    hi = p[(feats.index >= idx[850]) & (feats.index <= idx[1100])]
    lo = p[(feats.index < idx[800]) | (feats.index > idx[1200])]
    assert hi.mean() > lo.mean()


def test_stress_prob_is_causal_no_lookahead():
    # Truncation invariance = the defining property of a CAUSAL filter: removing
    # FUTURE observations must not change the filtered P(stress) at earlier times
    # (with the same frozen params). If Viterbi/smoothing leaked in, it would.
    spot, idx, stress = make_regime_spot()
    feats = reg.crash_features(spot, freq="M")
    T = str(idx[1150].date())                    # train window covers both regimes
    cut_date = idx[1200]                          # drop everything AFTER this
    p_full = reg.causal_stress_prob(feats, train_end=T)
    feats_trunc = feats[feats.index <= cut_date]  # training set (<=T) unchanged
    p_trunc = reg.causal_stress_prob(feats_trunc, train_end=T)
    both = p_trunc.index
    assert np.allclose(p_full.reindex(both).values, p_trunc.values, atol=1e-6)


def test_stress_prob_in_unit_interval():
    spot, idx, stress = make_regime_spot()
    feats = reg.crash_features(spot, freq="M")
    p = reg.causal_stress_prob(feats, train_end=str(idx[-1].date()))
    assert (p >= -1e-9).all() and (p <= 1 + 1e-9).all()


def test_fallback_when_hmmlearn_missing(monkeypatch):
    # force the ImportError path -> causal vol-percentile rule, still sane
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name.startswith("hmmlearn"):
            raise ImportError("no hmmlearn")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    spot, idx, stress = make_regime_spot()
    feats = reg.crash_features(spot, freq="M")
    p = reg.causal_stress_prob(feats, train_end=str(idx[-1].date()))
    assert (p >= 0).all() and (p <= 1).all()
    hi = p[(feats.index >= idx[850]) & (feats.index <= idx[1100])]
    assert hi.mean() > 0                     # flags the stress window


# --- Viterbi (validation) --------------------------------------------------
def test_viterbi_labels_stress_window():
    spot, idx, stress = make_regime_spot()
    feats = reg.crash_features(spot, freq="M")
    vit = reg.viterbi_states(feats, train_end=str(idx[-1].date()))
    assert set(vit.unique()).issubset({0, 1})
    in_window = vit[(feats.index >= idx[850]) & (feats.index <= idx[1100])]
    assert in_window.mean() > 0.5            # mostly labeled stress


# --- overlay ---------------------------------------------------------------
def test_regime_exposure_cuts_in_stress_and_lagged():
    spot, idx, stress = make_regime_spot()
    exp = reg.regime_exposure(spot, freq="M", train_end=str(idx[-1].date()))
    hi = exp[(exp.index >= idx[850]) & (exp.index <= idx[1100])]
    lo = exp[(exp.index < idx[800])]
    assert hi.mean() < lo.mean()             # de-risks during stress
    assert (exp >= 0).all() and (exp <= 1).all()


def test_run_regime_overlay_shape():
    spot, idx, stress = make_regime_spot()
    base = run_carry_backtest(spot, carry=_flat_carry(spot), freq="M",
                              n_long=2, n_short=2)
    res = reg.run_regime_overlay(base, spot, train_end=str(idx[900].date()))
    assert len(res["equity"]) > 0
    assert res["exposure"].notna().all()


# --- Path B: news risk-off tone feature ------------------------------------
def test_tone_adds_risk_tone_feature_col():
    spot, idx, stress = make_regime_spot()
    tone = make_regime_tone(spot, idx, stress)
    feats = reg.crash_features(spot, freq="M", tone_panel=tone)
    assert list(feats.columns)[:2] == ["fx_vol", "haven_spread"]   # fx_vol stays col 0
    assert "risk_tone" in feats.columns


def test_risk_tone_high_in_stress():
    spot, idx, stress = make_regime_spot()
    tone = make_regime_tone(spot, idx, stress)
    feats = reg.crash_features(spot, freq="M", tone_panel=tone)
    hi = feats["risk_tone"][(feats.index >= idx[850]) & (feats.index <= idx[1100])]
    lo = feats["risk_tone"][(feats.index < idx[800])]
    assert hi.mean() > lo.mean()          # tone negative in stress -> risk_tone up


def test_tone_feature_handles_tz_aware_panel():
    spot, idx, stress = make_regime_spot()
    tone = make_regime_tone(spot, idx, stress, tz=True)      # GDELT-style UTC index
    assert tone.index.tz is not None
    feats = reg.crash_features(spot, freq="M", tone_panel=tone)   # must not crash
    assert "risk_tone" in feats.columns


def test_tone_feature_no_lookahead_truncation_invariance():
    spot, idx, stress = make_regime_spot()
    tone = make_regime_tone(spot, idx, stress)
    feats = reg.crash_features(spot, freq="M", tone_panel=tone)
    T = str(idx[1150].date())
    cut = idx[1200]
    p_full = reg.causal_stress_prob(feats, train_end=T)
    p_trunc = reg.causal_stress_prob(feats[feats.index <= cut], train_end=T)
    both = p_trunc.index
    assert np.allclose(p_full.reindex(both).values, p_trunc.values, atol=1e-6)


def test_run_holdout_adds_tone_book():
    spot, carry = make_carry_world(n_days=1600)
    # build a tone panel matching the carry_world currencies
    m_idx = pd.date_range(spot.index[0], spot.index[-1], freq="M")
    tone = pd.DataFrame(np.random.default_rng(0).normal(0, 0.3,
                        (len(m_idx), spot.shape[1])), index=m_idx, columns=spot.columns)
    out = reg.run_holdout(spot, carry, train_end="2020-06-30", freq="M",
                          tone_panel=tone)
    assert set(out) == {"carry", "carry+regime", "carry+regime+tone"}


def test_run_holdout_reports_full_and_test():
    spot, carry = make_carry_world(n_days=1600)
    out = reg.run_holdout(spot, carry, train_end="2020-06-30", freq="M")
    assert set(out) == {"carry", "carry+regime"}
    for name in out:
        assert "full" in out[name] and "test" in out[name]


def _flat_carry(spot):
    """Simple monthly carry panel matching a spot frame's currencies."""
    m_idx = pd.date_range(spot.index[0], spot.index[-1], freq="M")
    levels = {c: (i - 4) for i, c in enumerate(spot.columns)}
    return pd.DataFrame({c: levels[c] for c in spot.columns}, index=m_idx)
