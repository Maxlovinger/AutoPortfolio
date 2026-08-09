"""
commodity_gate.py — the Tier-1 diversification GATE for commodities.

Two things get tested, because commodities offer two very different exposures:

  PASSIVE  : long a commodity basket / sector (DBC, oil, gold, ags...). Like the
             foreign gate — per proxy: Sharpe, corr to equity, tail corr, and the
             threshold rule (earns weight iff Sharpe_f > corr_f * Sharpe_equity).

  TREND    : a time-series-momentum BOOK across single commodities (long recent
             up-trends, short down-trends, no lookahead). This is the crisis-alpha
             candidate — the thing that historically RISES in equity selloffs.
             Tested as its own return stream, with the key CRISIS test: its mean
             return in the equity book's worst months (want > 0, not just low corr).

What Tier 1 does NOT test: pure cross-sectional carry (needs the futures curve) —
that is a Tier-2 question. Passive ETF returns already embed roll yield, so a
positive passive result would itself be partly a carry result.

Reuses cross_correlation's equity_monthly/currency_monthly and markowitz.py, so
results sit on the same monthly grid as the two-sleeve and foreign analyses.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import commodity_data as cd

OUT = os.path.join(os.path.dirname(__file__), "figures_commodity")
PPY = 12


# --- stats helpers ---------------------------------------------------------
def _sharpe(r: pd.Series, rf=0.04) -> float:
    r = r.dropna()
    if len(r) < 6:
        return np.nan
    mu, sd = r.mean() * PPY, r.std(ddof=1) * np.sqrt(PPY)
    return (mu - rf) / sd if sd > 0 else np.nan


def _tail_corr(a: pd.Series, b: pd.Series, anchor: pd.Series, q=0.20) -> float:
    df = pd.concat([a, b, anchor], axis=1).dropna()
    if len(df) < 12:
        return np.nan
    x, y, z = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    mask = z <= z.quantile(q)
    return x[mask].corr(y[mask]) if mask.sum() >= 4 else np.nan


def _tail_mean(a: pd.Series, anchor: pd.Series, q=0.20) -> float:
    """Mean of a in anchor's worst-q months — the crisis-alpha number."""
    df = pd.concat([a, anchor], axis=1).dropna()
    if len(df) < 12:
        return np.nan
    x, z = df.iloc[:, 0], df.iloc[:, 1]
    mask = z <= z.quantile(q)
    return x[mask].mean() if mask.sum() >= 4 else np.nan


# --- the trend-following book ----------------------------------------------
def trend_strategy(comm_rets: pd.DataFrame, lookback=12,
                   names=None) -> pd.Series:
    """
    Time-series momentum: each month hold +1 (long) if a commodity's trailing
    `lookback`-month return was positive, -1 (short) if negative; equal-weight
    across the commodities that have a signal. The signal is LAGGED (uses returns
    only through t-1) so month t's position never sees month t's return.

    Returns the monthly strategy return stream 'comm_trend'.
    """
    names = names or cd.TREND_BOOK
    R = comm_rets[[n for n in names if n in comm_rets.columns]].copy()
    if R.shape[1] == 0:
        return pd.Series(dtype=float, name="comm_trend")
    # trailing lookback-month compounded return, then shift 1 -> no lookahead
    mom = (1 + R).rolling(lookback).apply(np.prod, raw=True) - 1
    sig = np.sign(mom.shift(1))
    strat = (sig * R).mean(axis=1, skipna=True)      # equal-weight signed
    return strat.rename("comm_trend")


# --- data ------------------------------------------------------------------
def load_panel(start="2006-01-01"):
    """Monthly panel: deployed equity book, plain carry, every commodity ETF,
    and the constructed trend book. Per-pair dropna is used downstream."""
    from cross_correlation import equity_monthly, currency_monthly
    eq = equity_monthly()
    carry, _ = currency_monthly(start=start)
    carry.index = carry.index + pd.offsets.MonthEnd(0)
    comm = cd.download_returns(start=start)
    comm.index = comm.index + pd.offsets.MonthEnd(0)
    trend = trend_strategy(comm)
    panel = pd.concat([eq.rename("equity"), carry.rename("carry"), comm, trend],
                      axis=1)
    return panel


# --- passive gate table ----------------------------------------------------
def gate_table(panel: pd.DataFrame, rf=0.04) -> pd.DataFrame:
    """Per passive commodity proxy: Sharpe, corr to equity, tail corr (equity's
    worst 20%), corr to carry, and the threshold verdict."""
    eq, carry = panel["equity"], panel["carry"]
    sh_eq = _sharpe(eq, rf)
    rows = {}
    for grp, names in cd.COMMODITY_GROUPS.items():
        for name in names:
            if name not in panel:
                continue
            f = panel[name]
            pair = pd.concat([eq, f], axis=1).dropna()
            if len(pair) < 12:
                continue
            corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            sh_f = _sharpe(f, rf)
            rows[name] = {
                "group": grp,
                "sharpe": sh_f,
                "corr_equity": corr,
                "tail_corr_eq": _tail_corr(f, eq, eq),
                "corr_carry": pd.concat([f, carry], axis=1).dropna()
                                .pipe(lambda d: d.iloc[:, 0].corr(d.iloc[:, 1])),
                "n": len(pair),
                "threshold": corr * sh_eq,
                "earns_weight": bool((sh_f > corr * sh_eq) and (sh_f > 0)),
            }
    return pd.DataFrame(rows).T.sort_values("corr_equity")


# --- strategy gate (passive basket vs trend book) --------------------------
def strategy_gate(panel: pd.DataFrame, rf=0.04) -> pd.DataFrame:
    """
    The headline: treat the diversified-basket (DBC) and the TREND book as
    'assets' and score each vs the equity book — Sharpe, correlation, tail corr,
    and the CRISIS number (mean monthly return in equity's worst 20% months).
    """
    eq, carry = panel["equity"], panel["carry"]
    sh_eq = _sharpe(eq, rf)
    out = {}
    for name in ["Commodities", "comm_trend"]:
        if name not in panel:
            continue
        f = panel[name]
        pair = pd.concat([eq, f], axis=1).dropna()
        if len(pair) < 12:
            continue
        corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        sh_f = _sharpe(f, rf)
        out[name] = {
            "sharpe": sh_f,
            "corr_equity": corr,
            "tail_corr_eq": _tail_corr(f, eq, eq),
            "corr_carry": pd.concat([f, carry], axis=1).dropna()
                            .pipe(lambda d: d.iloc[:, 0].corr(d.iloc[:, 1])),
            "crisis_mean_ret": _tail_mean(f, eq),      # want > 0 for trend
            "eq_crisis_mean": _tail_mean(eq, eq),      # equity's own tail mean
            "threshold": corr * sh_eq,
            "earns_weight": bool((sh_f > corr * sh_eq) and (sh_f > 0)),
            "n": len(pair),
        }
    return pd.DataFrame(out).T


# --- 3-asset Markowitz -----------------------------------------------------
def markowitz_add(panel: pd.DataFrame, name: str, rf=0.04):
    """Long-only max-Sharpe over {equity, carry, name}, raw annualized moments
    (same convention as cross_correlation.markowitz_sleeve)."""
    import markowitz as mk
    cols = ["equity", "carry", name]
    df = panel[cols].dropna()
    if len(df) < 18:
        return None
    mu, Sig = df.mean().values * PPY, df.cov().values * PPY
    w = mk.max_sharpe(mu, Sig, rf=rf)
    base = panel[["equity", "carry"]].dropna()
    mu0, Sig0 = base.mean().values * PPY, base.cov().values * PPY
    w0 = mk.max_sharpe(mu0, Sig0, rf=rf)
    return {"weights": dict(zip(cols, w)), "sharpe": mk.sharpe_ratio(w, mu, Sig, rf),
            "foreign_weight": w[2], "base_sharpe": mk.sharpe_ratio(w0, mu0, Sig0, rf),
            "n": len(df)}


# --- figure ----------------------------------------------------------------
def make_figure(panel: pd.DataFrame, rf=0.04):
    os.makedirs(OUT, exist_ok=True)
    tab = gate_table(panel, rf)
    sg = strategy_gate(panel, rf)
    eq = panel["equity"]
    sh_eq = _sharpe(eq, rf)

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    valid = panel.dropna(how="all")
    span = f"{valid.index[0]:%Y-%m}..{valid.index[-1]:%Y-%m}"
    fig.suptitle(f"Commodity diversification gate (Tier 1, ETF proxies, {span})",
                 fontsize=15, fontweight="bold")

    colors = {"broad": "C2", "energy": "C3", "metals": "C4", "ags": "C5"}

    # (1) passive corr to US: full vs tail
    a = ax[0, 0]
    y = np.arange(len(tab))
    a.barh(y, tab["corr_equity"].astype(float),
           color=[colors[g] for g in tab["group"]], alpha=.85, label="full corr")
    a.scatter(tab["tail_corr_eq"].astype(float), y, color="k", s=28, zorder=5,
              label="tail corr (eq worst 20%)")
    a.set_yticks(y); a.set_yticklabels(tab.index, fontsize=8)
    a.axvline(0, color="k", lw=.8)
    a.set_title("Passive commodity corr to US equity")
    a.legend(fontsize=8); a.grid(alpha=.3, axis="x")

    # (2) trend book vs equity, crash months shaded
    a = ax[0, 1]
    if "comm_trend" in panel:
        d = panel[["equity", "comm_trend"]].dropna()
        a.plot((1 + d["equity"]).cumprod(), color="C0", label="equity")
        a.plot((1 + d["comm_trend"]).cumprod(), color="C1", label="commodity trend")
        thr = d["equity"].quantile(0.20)
        for t in d.index[d["equity"] <= thr]:
            a.axvspan(t - pd.offsets.MonthEnd(1), t, color="red", alpha=.08)
        a.set_yscale("log")
        a.set_title("Growth of $1 (red = equity's worst months)")
        a.legend(fontsize=9); a.grid(alpha=.3)

    # (3) crisis alpha: mean monthly return in equity's worst 20% months
    a = ax[1, 0]
    labels, vals, cols = [], [], []
    for nm, c in [("equity", "C0"), ("Commodities", "C2"), ("comm_trend", "C1")]:
        if nm in panel:
            labels.append(nm); cols.append(c)
            vals.append(_tail_mean(panel[nm], eq) * 100)
    a.bar(labels, vals, color=cols)
    a.axhline(0, color="k", lw=.8)
    a.set_title("Crisis alpha: mean monthly % in equity's worst 20% months")
    a.grid(alpha=.3, axis="y")
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:+.2f}%", ha="center", va="bottom" if v >= 0 else "top")

    # (4) verdict
    a = ax[1, 1]; a.axis("off")
    mk_tr = markowitz_add(panel, "comm_trend") if "comm_trend" in panel else None
    mk_pa = markowitz_add(panel, "Commodities") if "Commodities" in panel else None
    txt = f"ANCHOR  equity Sharpe {sh_eq:.2f}\n\n"
    for nm in ["Commodities", "comm_trend"]:
        if nm in sg.index:
            r = sg.loc[nm]
            txt += (f"{nm}:\n"
                    f"  Sharpe {r['sharpe']:.2f}  corr {r['corr_equity']:+.2f}  "
                    f"tailcorr {r['tail_corr_eq']:+.2f}\n"
                    f"  crisis mean ret {r['crisis_mean_ret']*100:+.2f}%/mo  "
                    f"earns_wt={bool(r['earns_weight'])}\n")
    if mk_pa:
        txt += (f"\nMarkowitz +passive : wt {mk_pa['foreign_weight']*100:.0f}%  "
                f"Sh {mk_pa['base_sharpe']:.2f}->{mk_pa['sharpe']:.2f}\n")
    if mk_tr:
        txt += (f"Markowitz +trend   : wt {mk_tr['foreign_weight']*100:.0f}%  "
                f"Sh {mk_tr['base_sharpe']:.2f}->{mk_tr['sharpe']:.2f}\n")
    txt += ("\nREAD: passive commodities are a low-Sharpe drag;\n"
            "the case rests on the TREND book's CRISIS return\n"
            "(positive in equity selloffs = true diversifier)\n"
            "and whether Markowitz gives it real weight.")
    a.text(0.02, 0.98, txt, va="top", family="monospace", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "commodity_gate.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


# --- CLI -------------------------------------------------------------------
if __name__ == "__main__":
    panel = load_panel()
    print("\n=== PASSIVE GATE TABLE ===")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(gate_table(panel).round(3))
    print("\n=== STRATEGY GATE (passive basket vs TREND book) ===")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(strategy_gate(panel).round(3))
    print("\n=== 3-ASSET MARKOWITZ ===")
    print("  +passive:", markowitz_add(panel, "Commodities"))
    print("  +trend  :", markowitz_add(panel, "comm_trend"))
    path = make_figure(panel)
    print(f"\nSaved gate figure -> {path}")
