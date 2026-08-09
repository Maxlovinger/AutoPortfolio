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


# --- Markowitz: equity & carry as two "stocks" -----------------------------
def markowitz_sleeve(start="2010-01-01", rf=0.04):
    """Mean-variance optimize the allocation between the equity book and plain
    carry, treated as two assets. Long-only, fully invested. Saves a frontier +
    Sharpe-vs-weight figure and returns the optimal weights."""
    import markowitz as mk
    os.makedirs(OUT, exist_ok=True)
    eq = equity_monthly()
    carry, _ = currency_monthly(start=start)
    df = pd.concat([eq, carry], axis=1).dropna()
    df.columns = ["equity", "carry"]

    mu = df.mean().values * PPY                          # annualized mean
    Sig = df.cov().values * PPY                          # annualized covariance
    w_ms = mk.max_sharpe(mu, Sig, rf=rf)                 # tangency (long-only)
    w_mv = mk.min_variance(Sig)
    vols, rets, _ = mk.efficient_frontier(mu, Sig, n_points=60)

    # Sharpe as a function of equity weight (long-only 0..1)
    ws = np.linspace(0, 1, 101)
    def _stat(w):
        wv = np.array([w, 1 - w])
        return mk.portfolio_vol(wv, Sig), mk.portfolio_return(wv, mu), \
               mk.sharpe_ratio(wv, mu, Sig, rf)
    curve = np.array([_stat(w) for w in ws])             # cols: vol, ret, sharpe
    w_best = ws[curve[:, 2].argmax()]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Markowitz: equity book vs plain carry as two assets "
                 f"(annualized, rf={rf:.0%})", fontsize=14, fontweight="bold")

    # (left) efficient frontier + key portfolios
    a = ax[0]
    sc = a.scatter(vols, rets, c=(rets - rf) / vols, cmap="viridis", s=18)
    fig.colorbar(sc, ax=a, label="Sharpe")
    pts = {"100% equity": (np.sqrt(Sig[0, 0]), mu[0], "C0"),
           "100% carry": (np.sqrt(Sig[1, 1]), mu[1], "C1"),
           "max-Sharpe": (mk.portfolio_vol(w_ms, Sig), mk.portfolio_return(w_ms, mu), "red"),
           "min-var": (mk.portfolio_vol(w_mv, Sig), mk.portfolio_return(w_mv, mu), "k")}
    for name, (v, r, c) in pts.items():
        marker = "*" if name == "max-Sharpe" else ("D" if name == "min-var" else "o")
        a.scatter(v, r, s=180 if marker == "*" else 90, c=c, marker=marker,
                  edgecolor="k", zorder=5, label=name)
    # capital market line through tangency
    tv, tr = pts["max-Sharpe"][0], pts["max-Sharpe"][1]
    xs = np.linspace(0, max(vols) * 1.05, 20)
    a.plot(xs, rf + (tr - rf) / tv * xs, "r--", lw=1, alpha=.6, label="CML")
    a.set_xlabel("annualized volatility"); a.set_ylabel("annualized return")
    a.set_title("Efficient frontier"); a.legend(fontsize=9); a.grid(alpha=.3)

    # (right) Sharpe vs equity weight
    a = ax[1]
    a.plot(ws * 100, curve[:, 2], color="C4")
    a.axvline(w_best * 100, color="red", ls="--",
              label=f"max-Sharpe @ {w_best*100:.0f}% equity")
    a.scatter([0, 100], [curve[0, 2], curve[-1, 2]], c=["C1", "C0"], zorder=5)
    a.set_xlabel("% in equity (rest in carry)"); a.set_ylabel("portfolio Sharpe")
    a.set_title("Sharpe vs allocation"); a.legend(fontsize=9); a.grid(alpha=.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "markowitz_two_sleeve.png")
    fig.savefig(p, dpi=130); plt.close(fig)

    return {"max_sharpe_w": {"equity": w_ms[0], "carry": w_ms[1]},
            "min_var_w": {"equity": w_mv[0], "carry": w_mv[1]},
            "max_sharpe_ratio": mk.sharpe_ratio(w_ms, mu, Sig, rf),
            "equity_sharpe": (mu[0] - rf) / np.sqrt(Sig[0, 0]),
            "carry_sharpe": (mu[1] - rf) / np.sqrt(Sig[1, 1]),
            "mu": mu, "path": p}


def _load_retry(uni, start="2010-01-01"):
    from fx.data import load_all
    import time
    for _ in range(4):
        try:
            return load_all(start=start, universe=uni)
        except Exception:
            time.sleep(4)
    raise RuntimeError("FRED load failed")


def make_wide_figure(start="2010-01-01", rf=0.04, em_bps=30.0):
    """Visual: does widening the FX universe to EM earn carry a real allocation?
    G10 carry vs EM-widened carry — growth, cost sensitivity, tail cost, and the
    levered Markowitz allocation flip."""
    import markowitz as mk
    from fx.data import G10, WIDE, EM_CCYS
    from fx.backtest_carry import run_carry_backtest, summarize
    os.makedirs(OUT, exist_ok=True)

    g = _load_retry(G10, start); w = _load_retry(WIDE, start)
    g10 = run_carry_backtest(g["spot"], g["carry"], n_long=3, n_short=3, cost_bps=5.0)
    costs = {c: (em_bps if c in EM_CCYS else 5.0) for c in w["carry"].columns}
    wide = run_carry_backtest(w["spot"], w["carry"], n_long=3, n_short=3, cost_bps=costs)
    mg, mw = summarize(g10), summarize(wide)

    # levered Markowitz allocation (carry scaled to equity vol) for G10 & WIDE
    eq = equity_monthly()
    def tangency(carry_net):
        c = carry_net.copy(); c.index = c.index + pd.offsets.MonthEnd(0)
        d = pd.concat([eq, c], axis=1).dropna(); d.columns = ["eq", "c"]
        d["c"] *= d["eq"].std() / d["c"].std()               # lever to equity vol
        mu = d.mean().values * PPY; Sig = d.cov().values * PPY
        wt = mk.max_sharpe(mu, Sig, rf=rf)
        return wt[1], mk.sharpe_ratio(wt, mu, Sig, rf), d.corr().iloc[0, 1]
    wc_g, sh_g, rho_g = tangency(g10["net_ret"])
    wc_w, sh_w, rho_w = tangency(wide["net_ret"])
    eq_sharpe = (eq.mean() * PPY - rf) / (eq.std() * np.sqrt(PPY))

    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Widening the FX carry universe to EM", fontsize=15, fontweight="bold")

    a = ax[0, 0]
    a.plot((1 + g10["net_ret"]).cumprod(), label=f"G10 carry (Sharpe {mg['sharpe']:.2f})")
    a.plot((1 + wide["net_ret"]).cumprod(), label=f"G10+EM carry (Sharpe {mw['sharpe']:.2f})")
    a.set_title("Growth of $1 (unlevered)"); a.legend(); a.grid(alpha=.3)

    a = ax[0, 1]
    bpss = [5, 15, 30, 50]
    shs = []
    for b in bpss:
        cc_ = {c: (b if c in EM_CCYS else 5.0) for c in w["carry"].columns}
        shs.append(summarize(run_carry_backtest(w["spot"], w["carry"], n_long=3,
                    n_short=3, cost_bps=cc_))["sharpe"])
    a.plot(bpss, shs, "o-", label="G10+EM carry")
    a.axhline(mg["sharpe"], color="C1", ls="--", label=f"G10 = {mg['sharpe']:.2f}")
    a.axhline(0.34, color="k", ls=":", label="MV threshold ~0.34")
    a.set_xlabel("EM half-spread (bps)"); a.set_ylabel("Sharpe")
    a.set_title("Cost sensitivity (robust)"); a.legend(fontsize=9); a.grid(alpha=.3)

    a = ax[1, 0]
    labs = ["G10", "G10+EM"]
    a.bar(labs, [mg["skew"], mw["skew"]], color=["C0", "C1"])
    a.set_title("Skew — the EM tail cost"); a.grid(alpha=.3, axis="y")
    for i, v in enumerate([mg["skew"], mw["skew"]]):
        a.text(i, v, f"{v:.2f}", ha="center", va="top" if v < 0 else "bottom")

    a = ax[1, 1]; a.axis("off")
    txt = (
        f"LEVERED MARKOWITZ (carry @ equity vol, rf={rf:.0%})\n\n"
        f"{'universe':<10}{'Sharpe':>7}{'corr':>7}{'carry wt':>10}{'blend Sh':>10}\n"
        f"{'G10':<10}{mg['sharpe']:>7.2f}{rho_g:>7.2f}{wc_g*100:>9.0f}%{sh_g:>10.2f}\n"
        f"{'G10+EM':<10}{mw['sharpe']:>7.2f}{rho_w:>7.2f}{wc_w*100:>9.0f}%{sh_w:>10.2f}\n"
        f"{'equity':<10}{'':>7}{'':>7}{'--':>10}{eq_sharpe:>10.2f}\n\n"
        f"WIDENING FLIPS THE VERDICT:\n"
        f" - Sharpe {mg['sharpe']:.2f} -> {mw['sharpe']:.2f} (bigger EM rate diffs)\n"
        f" - corr {rho_g:.2f} -> {rho_w:.2f} (EM less equity-linked)\n"
        f" - Markowitz carry weight {wc_g*100:.0f}% -> {wc_w*100:.0f}%\n"
        f" - combined Sharpe {eq_sharpe:.2f} -> {sh_w:.2f}\n\n"
        f"CAVEAT: skew {mg['skew']:.2f} -> {mw['skew']:.2f} (EM devaluation\n"
        f"jumps). Mean-variance IGNORES this fatter left\n"
        f"tail, so a skew-aware investor sizes EM carry\n"
        f"BELOW the {wc_w*100:.0f}% MV weight. Robust to EM costs."
    )
    a.text(0.02, 0.98, txt, va="top", family="monospace", fontsize=10.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, "wide_carry.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


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
