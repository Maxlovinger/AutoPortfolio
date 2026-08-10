"""
sleeve_markowitz.py — "before we dismiss it, what does the optimizer want?"

Runs Markowitz over the INDIVIDUAL assets of a sleeve (each commodity / each
currency as its own asset) to find the optimal distribution, for both:
  * COMMODITIES — the clean 13-market Databento front returns (2010-2026)
  * CURRENCIES  — G10+EM carry-inclusive monthly excess returns

CRUCIAL discipline (this project has repeatedly shown max-Sharpe is an in-sample
MIRAGE): we report the in-sample optimum BUT the verdict rests on an OUT-OF-SAMPLE
split — estimate weights on the first 60% of months, freeze them, apply to the
last 40%. If the "optimal" distribution only shines in-sample and dies OOS, that
confirms the dismissal; if it holds up OOS, that's a real reason not to dismiss.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import markowitz as mk

PPY = 12


def _ann(r):
    return r.mean().values * PPY, r.cov().values * PPY


def _sharpe(series):
    s = series.dropna()
    sd = s.std(ddof=1)
    return (s.mean() / sd * np.sqrt(PPY)) if sd > 0 else np.nan


def _oos_split(rets, frac=0.6, rf=0.0):
    """Estimate max-Sharpe weights on the first `frac` of months, freeze, apply to
    the rest. Returns (oos_sharpe_maxsharpe, oos_sharpe_minvar, oos_sharpe_eqw)."""
    n = len(rets)
    k = int(n * frac)
    train, test = rets.iloc[:k], rets.iloc[k:]
    mu, Sig = _ann(train)
    w_ms = mk.max_sharpe(mu, Sig, rf=rf, allow_short=True)
    w_mv = mk.min_variance(Sig, allow_short=True)
    w_eq = np.repeat(1 / rets.shape[1], rets.shape[1])
    oos = {}
    for name, w in [("max_sharpe", w_ms), ("min_var", w_mv), ("equal", w_eq)]:
        oos[name] = _sharpe((test * w).sum(axis=1))
    return oos, w_ms


def analyze(rets: pd.DataFrame, label: str, rf=0.0, benchmark=None):
    rets = rets.dropna(how="all").dropna(axis=1, how="all")
    rets = rets.dropna()                         # common overlap for a clean cov
    print(f"\n{'='*68}\n{label}: {rets.shape[1]} assets, "
          f"{rets.index[0]:%Y-%m}..{rets.index[-1]:%Y-%m} ({len(rets)} months, common overlap)")
    mu, Sig = _ann(rets)

    # in-sample optima (the mirage — flagged)
    w_ms = mk.max_sharpe(mu, Sig, rf=rf, allow_short=True)
    w_mv = mk.min_variance(Sig, allow_short=True)
    is_ms = (mu @ w_ms - rf) / np.sqrt(w_ms @ Sig @ w_ms)
    is_mv = (mu @ w_mv - rf) / np.sqrt(w_mv @ Sig @ w_mv)
    print(f"  IN-SAMPLE  max-Sharpe {is_ms:.2f} (gross {np.abs(w_ms).sum():.1f}x, "
          f"leverage) | min-var {is_mv:.2f}")

    # the optimal distribution (top longs / shorts)
    w = pd.Series(w_ms, index=rets.columns).sort_values()
    print("  optimal (max-Sharpe) tilt — top shorts / longs:")
    print("    shorts:", ", ".join(f"{c} {w[c]:+.2f}" for c in w.index[:4]))
    print("    longs :", ", ".join(f"{c} {w[c]:+.2f}" for c in w.index[-4:]))

    # OUT-OF-SAMPLE (the honest verdict)
    oos, _ = _oos_split(rets, frac=0.6, rf=rf)
    print(f"  OUT-OF-SAMPLE (train 60% -> test 40%): "
          f"max-Sharpe {oos['max_sharpe']:+.2f} | min-var {oos['min_var']:+.2f} "
          f"| equal-weight {oos['equal']:+.2f}")
    if benchmark is not None:
        print(f"  deployed book (benchmark) Sharpe: {benchmark:.2f}")
    verdict = ("OOS optimum HOLDS -> reconsider" if oos["max_sharpe"] > 0.4
               else "OOS optimum COLLAPSES -> in-sample mirage, dismissal stands")
    print(f"  => {verdict}")
    return {"is_max_sharpe": is_ms, "oos": oos, "weights": w}


def commodities():
    import databento_curve as dbc
    rets = dbc.front_returns(dbc.download(use_cache=True))
    return analyze(rets, "COMMODITIES (Databento front returns)")


def currencies(start="2010-01-01"):
    from fx.data import load_all, WIDE
    from fx.backtest_carry import build_asset_returns, run_carry_backtest, summarize
    d = None
    import time
    for _ in range(4):
        try:
            d = load_all(start=start, universe=WIDE); break
        except Exception:
            time.sleep(4)
    asset_ret, _ = build_asset_returns(d["spot"], d["carry"], freq="M")
    # benchmark: the deployed rank-based carry book on the same universe
    bench = summarize(run_carry_backtest(d["spot"], d["carry"], n_long=3, n_short=3,
                                         cost_bps=5.0))["sharpe"]
    return analyze(asset_ret, "CURRENCIES (G10+EM carry-inclusive)", benchmark=bench)


if __name__ == "__main__":
    commodities()
    currencies()
    print("\nNote: 'optimal' in-sample weights use historical means, which are pure "
          "noise out-of-sample. The OOS row is the one that matters.")
