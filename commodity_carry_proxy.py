"""
commodity_carry_proxy.py — a FREE carry signal from the spot-vs-front BASIS.

IBKR can't serve expired contracts, so we can't reconstruct true front/next carry
history from it (see the probe). This builds an honest, free PROXY instead:

    basis_i,t = log(spot_i,t) - log(front_future_i,t)

using FRED/IMF monthly SPOT commodity prices (free, deep) and the front-month
future we already ingested (commodity_futures continuous panel). Backwardation
(cash above the front future) -> positive roll yield -> a long-carry candidate.

IMPORTANT honest caveats baked into the design:
  * The IMF spot and the exchange future are quoted in DIFFERENT units/grades, so
    the basis has a constant per-commodity OFFSET. We therefore STANDARDIZE each
    commodity's basis against its own history (no-lookahead expanding z-score,
    `zscore_vs_history`), which removes the offset and leaves the time-varying
    term-structure state. So this is a carry-TIMING signal (is this market's curve
    unusually backwardated vs its own norm), not a clean absolute cross-market
    carry level. It is a proxy — to be confirmed later on true curve data
    (Databento free credit / a vendor).
  * Monthly only; IMF spot is a monthly average vs a month-end future -> some noise.

Everything is validated the usual way: the signal is null-tested through the
commodity_signals harness BEFORE any backtest is trusted.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from utils import MONTH_END
from commodity_factors import zscore_vs_history
import commodity_trend as ctr

CACHE = os.path.join(os.path.dirname(__file__), ".commodity_cache", "fred_spot.pkl")

# FRED monthly SPOT series (IMF/World Bank "Global price of X"), verified present.
FRED_SPOT = {
    "Oil": "POILWTIUSDM", "NatGas": "PNGASUSUSDM", "Copper": "PCOPPUSDM",
    "Corn": "PMAIZMTUSDM", "Wheat": "PWHEAMTUSDM", "Soybeans": "PSOYBUSDM",
    "SoyOil": "PSOILUSDM", "Sugar": "PSUGAISAUSDM", "Coffee": "PCOFFOTMUSDM",
    "Cotton": "PCOTTINDUSDM",
}


# --- data ------------------------------------------------------------------
def fetch_spot(start="2010-01-01", use_cache=True) -> pd.DataFrame:
    """Monthly spot prices per commodity from FRED (month-end grid). Cached."""
    if use_cache and os.path.exists(CACHE):
        return pd.read_pickle(CACHE)
    from fx.data import _fred_client
    fred = _fred_client()
    cols = {}
    for name, sid in FRED_SPOT.items():
        s = fred.get_series(sid, observation_start=start).dropna()
        s.index = pd.to_datetime(s.index)
        cols[name] = s.resample(MONTH_END).last()
    df = pd.DataFrame(cols)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_pickle(CACHE)
    return df


def build_basis(spot: pd.DataFrame, front: pd.DataFrame) -> pd.DataFrame:
    """log(spot) - log(front) on the shared markets & monthly grid."""
    cols = [c for c in spot.columns if c in front.columns]
    idx = spot.index.union(front.index)
    sp = np.log(spot[cols].reindex(idx))
    fr = np.log(front[cols].reindex(idx))
    return (sp - fr).dropna(how="all")


def carry_signal(basis: pd.DataFrame, min_periods=12) -> pd.DataFrame:
    """No-lookahead carry signal: each commodity's basis z-scored vs its own
    history (removes the unit offset; high = unusually backwardated = long)."""
    return zscore_vs_history(basis, min_periods=min_periods)


def load_signal(start="2010-01-01", use_cache=True):
    """Convenience: (carry_signal panel, front-future monthly returns) aligned."""
    import commodity_futures as cfut
    spot = fetch_spot(start=start, use_cache=use_cache)
    front = cfut.load_continuous(monthly=True)
    basis = build_basis(spot, front)
    sig = carry_signal(basis)
    rets = front.pct_change()
    return sig, rets, basis


# --- carry backtest (cross-sectional, vol-targeted) ------------------------
def carry_weights(signal: pd.DataFrame, rets: pd.DataFrame,
                  vol_lookback=12) -> pd.DataFrame:
    """
    Cross-sectional carry book: scale the (already-lagged) signal by inverse vol,
    then DEMEAN across markets each month so the vol-scaled book is ~market-neutral
    (long above-average carry, short below), then risk-normalize to gross |w| = 1.
    Demeaning AFTER vol-scaling is what preserves neutrality. No-lookahead.
    """
    iv = ctr.inv_vol(rets, vol_lookback)
    raw = signal * iv
    raw = raw.sub(raw.mean(axis=1), axis=0)            # neutralize the vol-scaled book
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    return raw.div(gross, axis=0)


def backtest(signal, rets, target_vol=0.12, cap=2.0, cost_bps=3.0,
             vol_target=True) -> dict:
    """Vol-targeted cross-sectional carry book (same overlay machinery as trend)."""
    w = carry_weights(signal, rets)
    gross = (w * rets).sum(axis=1, min_count=1)
    dw = (w - w.shift(1)).abs().sum(axis=1)
    cost = dw * cost_bps / 1e4
    if vol_target:
        book_vol = gross.rolling(12).std() * np.sqrt(12)
        e = (target_vol / book_vol).clip(upper=cap).shift(1)
    else:
        e = pd.Series(1.0, index=gross.index)
    net = (e * gross - cost).dropna()
    return {"net": net, "gross": gross, "weights": w, "turnover": dw}


def combine(carry_net: pd.Series, trend_net: pd.Series,
            target_vol=0.12) -> pd.Series:
    """Equal-RISK blend of the carry and trend books (each scaled to target vol),
    the base commodity sleeve."""
    def _scale(r):
        sd = r.std(ddof=1) * np.sqrt(12)
        return r * (target_vol / sd) if sd > 0 else r
    df = pd.concat([_scale(carry_net), _scale(trend_net)], axis=1).dropna()
    return 0.5 * df.iloc[:, 0] + 0.5 * df.iloc[:, 1]


# --- CLI -------------------------------------------------------------------
if __name__ == "__main__":
    import commodity_signals as cs
    sig, rets, basis = load_signal()
    print(f"Carry proxy: {sig.shape[1]} markets, basis "
          f"{basis.dropna(how='all').index.min():%Y-%m}.."
          f"{basis.dropna(how='all').index.max():%Y-%m}\n")

    # 1) is the carry proxy even predictive? (null-tested, like every signal)
    print("=== CARRY-PROXY SIGNAL TEST vs next-month futures returns ===")
    ev = cs.evaluate_signal(sig, rets, name="carry_proxy", n_null=300)
    for k in ("cols", "pooled_ic", "pooled_ic_z", "ts_sharpe", "xs_mean_ic",
              "xs_ic_z", "xs_pctile", "passes"):
        if k in ev:
            print(f"  {k:<14}{ev[k]}")

    # 2) the carry backtest + combine with trend
    car = backtest(sig, rets)
    trd = ctr.backtest(ctr.load_returns())
    comb = combine(car["net"], trd["net"])
    print(f"\n{'book':<22}{'Sharpe':>8}{'CAGR':>8}{'Vol':>7}{'MaxDD':>8}{'Skew':>7}{'n':>5}")
    for name, r in [("carry proxy", car["net"]), ("trend", trd["net"]),
                    ("carry+trend (blend)", comb)]:
        m = ctr.performance(r)
        print(f"{name:<22}{m['sharpe']:>8.2f}{m['cagr']*100:>7.1f}%"
              f"{m['vol']*100:>6.1f}%{m['maxdd']*100:>7.1f}%{m['skew']:>7.2f}{m['n']:>5}")

    ca = ctr.crisis_alpha(comb)
    if ca.get("n", 0) >= 12:
        print(f"\nCRISIS ALPHA of carry+trend blend ({ca['n']} mo): "
              f"corr {ca['corr']:+.2f}, in equity's worst {ca['n_stress']} mo "
              f"{ca['trend_mean_stress']*100:+.2f}%/mo")
