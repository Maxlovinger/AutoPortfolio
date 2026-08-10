"""
commodity_book.py — the REAL commodity sleeve on 16 years of true curve data.

Assembles carry + trend on the Databento front/second curve (13 CME-group
markets, 2010-2026), so the carry leg uses the TRUE front-vs-second slope (not
the FRED spot-basis proxy) and both legs run over a full commodity cycle incl.
the 2014-2020 bear.

  returns : clean front returns incl. roll yield (databento_curve.front_returns)
  carry   : raw annualized front-second slope LEVEL (comparable units now, so the
            standard cross-sectional carry factor — long backwardated / short
            contangoed — no z-score hack needed)
  trend   : the same vol-targeted TS-momentum book as before, on these returns

Reuses the vol-target/backtest machinery from commodity_trend + commodity_carry_proxy
and the null-test harness from commodity_signals.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import databento_curve as dbc
import commodity_trend as ctr
import commodity_carry_proxy as cp
import commodity_signals as cs
from commodity_factors import zscore_vs_history

OUT = os.path.join(os.path.dirname(__file__), "figures_commodity")


def load():
    """(rets, carry_level) monthly on the databento curve."""
    raw = dbc.download(use_cache=True)
    rets = dbc.front_returns(raw)
    front, second = dbc.curve_panels(raw)
    carry = dbc.carry_monthly(front, second)
    cols = [c for c in rets.columns if c in carry.columns]
    return rets[cols], carry[cols]


def run():
    rets, carry = load()
    span = f"{rets.dropna(how='all').index.min():%Y-%m}..{rets.dropna(how='all').index.max():%Y-%m}"
    print(f"Real curve: {rets.shape[1]} markets, {span} ({len(rets)} months)\n")

    # 1) is TRUE carry predictive? raw level AND z-scored, null-tested
    print("=== CARRY SIGNAL TEST (true front-second) vs next-month returns ===")
    for label, sig in [("carry_level", carry),
                       ("carry_zscore", zscore_vs_history(carry))]:
        ev = cs.evaluate_signal(sig, rets, name=label, n_null=300)
        print(f"  {label:<13} pooled_ic {ev['pooled_ic']:+.3f} z {ev['pooled_ic_z']:+.2f}"
              f" | xs_ic_z {ev.get('xs_ic_z', float('nan')):+.2f}"
              f" pctile {ev.get('xs_pctile', float('nan')):.2f} | passes={ev['passes']}")

    # 2) backtests: carry (raw level, lagged), trend, and the equal-risk blend
    car = cp.backtest(carry.shift(1), rets)
    trd = ctr.backtest(rets)
    comb = cp.combine(car["net"], trd["net"])

    print(f"\n{'book':<22}{'Sharpe':>8}{'CAGR':>8}{'Vol':>7}{'MaxDD':>8}{'Skew':>7}{'n':>5}")
    rows = [("carry (true)", car["net"]), ("trend", trd["net"]),
            ("carry+trend blend", comb)]
    for name, r in rows:
        m = ctr.performance(r)
        print(f"{name:<22}{m['sharpe']:>8.2f}{m['cagr']*100:>7.1f}%"
              f"{m['vol']*100:>6.1f}%{m['maxdd']*100:>7.1f}%{m['skew']:>7.2f}{m['n']:>5}")

    ca = ctr.crisis_alpha(comb)
    if ca.get("n", 0) >= 12:
        print(f"\nCRISIS ALPHA of blend ({ca['n']} mo vs equity): corr {ca['corr']:+.2f}, "
              f"in equity's worst {ca['n_stress']} mo {ca['trend_mean_stress']*100:+.2f}%/mo "
              f"(equity {ca['eq_mean_stress']*100:+.2f}%/mo)")

    # sub-period robustness: does carry survive the 2014-2020 bear?
    print("\nSUB-PERIOD carry Sharpe (does it survive the down-cycle?):")
    for lo, hi in [("2010", "2014"), ("2014", "2020"), ("2020", "2027")]:
        seg = car["net"].loc[lo:hi]
        m = ctr.performance(seg)
        print(f"  {lo}-{hi}: Sharpe {m['sharpe']:>5.2f}  (n={m['n']})")

    _figure(car["net"], trd["net"], comb, span)
    return car, trd, comb


def _figure(carry_net, trend_net, comb, span):
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle(f"Real commodity sleeve — true curve, {span}", fontweight="bold")
    a = ax[0]
    for s, c, lab in [(carry_net, "C0", "carry"), (trend_net, "C1", "trend"),
                      (comb, "C2", "carry+trend")]:
        a.plot((1 + s).cumprod(), color=c, label=lab)
    a.set_yscale("log"); a.set_title("Growth of $1"); a.legend(); a.grid(alpha=.3)
    a = ax[1]
    e = (1 + comb).cumprod()
    a.fill_between(e.index, (e / e.cummax() - 1).values, 0, color="C3", alpha=.5)
    a.set_title("carry+trend drawdown"); a.grid(alpha=.3)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, "commodity_book_true.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"\nSaved figure -> {p}")


if __name__ == "__main__":
    run()
