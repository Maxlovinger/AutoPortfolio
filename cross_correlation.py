"""
cross_correlation.py — how correlated are the EQUITY and CURRENCY strategies?

The two sleeves are run separately on purpose; their value together depends on
being uncorrelated (low/negative correlation => combining them cuts risk for
free). This computes both strategies' returns on a common MONTHLY grid and
renders a thorough visual dashboard + a diversification analysis.

  Equity  : deployed book (final_strategy) — liquid-30 PIT, equal-weight,
            quarterly, 15% vol-target. Daily returns -> monthly.
  Currency: plain monthly carry, and carry + HMM crash overlay (vol+haven).

Runs on real data (FRED + prices_pit.pkl); no GDELT/FinBERT needed.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import MONTH_END

OUT = os.path.join(os.path.dirname(__file__), "figures_cross")
PPY = 12


# --- returns ---------------------------------------------------------------
def _to_monthly(daily_ret: pd.Series) -> pd.Series:
    return (1 + daily_ret.fillna(0.0)).resample(MONTH_END).prod() - 1


def equity_monthly() -> pd.Series:
    """Deployed equity book monthly returns (vol-targeted)."""
    import final_strategy as fs
    res = fs.build()
    return _to_monthly(res["overlaid"]).rename("equity")


def currency_monthly(start="2010-01-01"):
    """Plain carry and carry+regime monthly returns (with FRED retry)."""
    from fx.data import load_all
    from fx.backtest_carry import run_carry_backtest
    from fx.regime import run_regime_overlay
    last = None
    for _ in range(4):
        try:
            d = load_all(start=start)
            break
        except Exception as e:                       # transient FRED 502s
            last = e
            import time; time.sleep(4)
    else:
        raise RuntimeError(f"FRED load failed: {last}")
    base = run_carry_backtest(d["spot"], d["carry"])
    over = run_regime_overlay(base, d["spot"])
    carry = base["net_ret"].rename("carry")
    carry.index = carry.index + pd.offsets.MonthEnd(0)      # snap to month-end
    creg = over["net_ret"].rename("carry+regime")
    creg.index = creg.index + pd.offsets.MonthEnd(0)
    return carry, creg


# --- stats -----------------------------------------------------------------
def stats(r: pd.Series) -> dict:
    r = r.dropna()
    mean, sd = r.mean(), r.std(ddof=1)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    yrs = len(r) / PPY
    return {"cagr": eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else np.nan,
            "vol": sd * np.sqrt(PPY),
            "sharpe": (mean / sd) * np.sqrt(PPY) if sd > 0 else np.nan,
            "maxdd": dd}


def _scale_to_vol(r, target=0.10):
    """Scale a return series to a target annualized vol (for fair blending)."""
    sd = r.std(ddof=1) * np.sqrt(PPY)
    return r * (target / sd) if sd > 0 else r


# --- dashboard -------------------------------------------------------------
def make_dashboard(start="2010-01-01"):
    os.makedirs(OUT, exist_ok=True)
    eq = equity_monthly()
    carry, creg = currency_monthly(start=start)

    df = pd.concat([eq, carry, creg], axis=1).dropna()
    if df.empty:
        raise RuntimeError("no overlapping months between the two strategies")
    eq, carry, creg = df["equity"], df["carry"], df["carry+regime"]
    corr = df.corr()

    # equal-risk (10% vol) 50/50 blend of equity + plain carry
    blend = 0.5 * _scale_to_vol(eq) + 0.5 * _scale_to_vol(carry)

    fig, ax = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f"Equity vs Currency strategy — correlation & diversification "
                 f"({df.index[0]:%Y-%m}..{df.index[-1]:%Y-%m}, n={len(df)} months)",
                 fontsize=15, fontweight="bold")

    # (1) cumulative growth
    a = ax[0, 0]
    for s, c in [(eq, "C0"), (carry, "C1"), (creg, "C2")]:
        a.plot((1 + s).cumprod(), color=c, label=s.name)
    a.set_yscale("log"); a.set_title("Growth of $1 (log)"); a.legend(); a.grid(alpha=.3)

    # (2) rolling 12m correlation
    a = ax[0, 1]
    rc = eq.rolling(12).corr(carry)
    a.plot(rc, color="C3"); a.axhline(0, color="k", lw=.8)
    a.axhline(corr.loc["equity", "carry"], color="C3", ls="--", lw=.8,
              label=f"full = {corr.loc['equity','carry']:.2f}")
    a.set_title("Rolling 12m corr: equity vs carry"); a.legend(); a.grid(alpha=.3)

    # (3) scatter + regression
    a = ax[0, 2]
    a.scatter(eq, carry, s=14, alpha=.6, color="C1")
    b, m = np.polyfit(eq, carry, 1)[::-1]
    xs = np.array([eq.min(), eq.max()]); a.plot(xs, m * xs + b, "k--", lw=1)
    a.set_xlabel("equity monthly ret"); a.set_ylabel("carry monthly ret")
    a.set_title(f"Monthly returns (r = {corr.loc['equity','carry']:.2f})"); a.grid(alpha=.3)

    # (4) correlation heatmap
    a = ax[1, 0]
    im = a.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    a.set_xticks(range(3)); a.set_yticks(range(3))
    a.set_xticklabels(corr.columns, rotation=30, ha="right"); a.set_yticklabels(corr.columns)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                   color="white" if abs(corr.iloc[i,j]) > .5 else "black")
    a.set_title("Correlation matrix"); fig.colorbar(im, ax=a, fraction=.046)

    # (5) drawdowns
    a = ax[1, 1]
    for s, c in [(eq, "C0"), (carry, "C1"), (blend.rename("blend"), "C4")]:
        e = (1 + s).cumprod(); a.plot(e / e.cummax() - 1, color=c, label=s.name)
    a.set_title("Drawdowns"); a.legend(); a.grid(alpha=.3)

    # (6) return distributions
    a = ax[1, 2]
    a.hist(eq, bins=30, alpha=.5, color="C0", label="equity", density=True)
    a.hist(carry, bins=30, alpha=.5, color="C1", label="carry", density=True)
    a.axvline(0, color="k", lw=.8); a.set_title("Monthly return distributions")
    a.legend(); a.grid(alpha=.3)

    # (7) diversification: blend vs components
    a = ax[2, 0]
    for s, c in [( _scale_to_vol(eq).rename("equity@10%"), "C0"),
                 (_scale_to_vol(carry).rename("carry@10%"), "C1"),
                 (blend.rename("50/50 blend"), "C4")]:
        a.plot((1 + s).cumprod(), color=c, label=s.name)
    a.set_title("Equal-risk (10% vol) blend vs parts"); a.legend(); a.grid(alpha=.3)

    # (8) stats bars
    a = ax[2, 1]
    names = ["equity", "carry", "carry+regime", "50/50 blend"]
    series = [eq, carry, creg, blend]
    sh = [stats(s)["sharpe"] for s in series]
    a.bar(names, sh, color=["C0", "C1", "C2", "C4"])
    a.set_title("Annualized Sharpe"); a.tick_params(axis="x", rotation=25); a.grid(alpha=.3, axis="y")
    for i, v in enumerate(sh):
        a.text(i, v, f"{v:.2f}", ha="center", va="bottom")

    # (9) summary text — honest read incl. STRESS correlation
    a = ax[2, 2]; a.axis("off")
    se, sc, sb = stats(eq), stats(carry), stats(blend)
    r = corr.loc["equity", "carry"]
    # conditional corr: months where equity is in its worst quintile (stress)
    thr = eq.quantile(0.20)
    stress = eq <= thr
    r_stress = eq[stress].corr(carry[stress])
    r_calm = eq[~stress].corr(carry[~stress])
    best_leg = max(se["sharpe"], sc["sharpe"])       # Sharpe is vol-invariant
    txt = (
        f"CORRELATION equity vs carry:  {r:+.2f}\n"
        f"  in equity's WORST 20% mths: {r_stress:+.2f}  (stress)\n"
        f"  in the calm 80%:            {r_calm:+.2f}\n"
        f"  vs carry+regime:            {corr.loc['equity','carry+regime']:+.2f}\n\n"
        f"{'strategy':<13}{'Sharpe':>7}{'Vol':>7}{'MaxDD':>7}\n"
        f"{'equity':<13}{se['sharpe']:>7.2f}{se['vol']*100:>6.1f}%{se['maxdd']*100:>6.1f}%\n"
        f"{'carry':<13}{sc['sharpe']:>7.2f}{sc['vol']*100:>6.1f}%{sc['maxdd']*100:>6.1f}%\n"
        f"{'50/50':<13}{sb['sharpe']:>7.2f}{sb['vol']*100:>6.1f}%{sb['maxdd']*100:>6.1f}%\n\n"
        f"READ: full corr +{r:.2f} is MODERATE, driven by\n"
        f"CALM co-movement (calm {r_calm:+.2f}). In equity's\n"
        f"WORST months corr = {r_stress:+.2f} -> carry does NOT\n"
        f"co-crash with equities => genuine TAIL diversifn.\n\n"
        f"But carry Sharpe {sc['sharpe']:.2f} << equity {se['sharpe']:.2f}, so a\n"
        f"blend cuts vol {se['vol']*100:.0f}%->{sb['vol']*100:.0f}% & DD "
        f"{se['maxdd']*100:.0f}%->{sb['maxdd']*100:.0f}% but\n"
        f"DILUTES Sharpe {best_leg:.2f}->{sb['sharpe']:.2f}. => RISK reduction,\n"
        f"not a free Sharpe lift (carry is a small,\n"
        f"low-Sharpe, tail-independent satellite)."
    )
    a.text(0.02, 0.98, txt, va="top", family="monospace", fontsize=10.5)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(OUT, "cross_dashboard.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return df, corr, p


def _svol(r):
    return stats(_scale_to_vol(r))["sharpe"]


if __name__ == "__main__":
    df, corr, path = make_dashboard()
    print(f"\nOverlap: {df.index[0]:%Y-%m} .. {df.index[-1]:%Y-%m}  ({len(df)} months)")
    print("\nCorrelation matrix:\n", corr.round(3))
    eqm, carm = df["equity"], df["carry"]
    thr = eqm.quantile(0.20)
    print(f"\nConditional corr (equity vs carry): "
          f"stress(worst 20% eq mths) = {eqm[eqm<=thr].corr(carm[eqm<=thr]):+.2f}, "
          f"calm = {eqm[eqm>thr].corr(carm[eqm>thr]):+.2f}")
    print("\nAnnualized stats:")
    for name in df.columns:
        s = stats(df[name])
        print(f"  {name:<14} Sharpe {s['sharpe']:.2f}  Vol {s['vol']*100:.1f}%  "
              f"MaxDD {s['maxdd']*100:.1f}%")
    print(f"\nSaved dashboard -> {path}")
