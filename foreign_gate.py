"""
foreign_gate.py — the Tier-1 diversification GATE for foreign equity.

Before building any foreign single-stock engine, we answer one question with
cheap ETF-proxy index data (foreign_data.py): does foreign equity earn a place
next to the US-equity book and the carry sleeve, or is it redundant?

Three tests, straight from the design doc:

  1. CORRELATION  — each foreign proxy vs the US equity book, overall AND in the
     equity book's worst months (tail corr is what actually matters).
  2. DIVERSIFICATION THRESHOLD — a proxy earns weight only if its Sharpe beats
     the bar set by its correlation:  Sharpe_f > corr_f * Sharpe_equity.
     Plus a 3-asset Markowitz (equity, carry, foreign) to see the real weight.
  3. EM SHARED-TAIL — does EM-country equity crash WITH EM carry? If the peso
     equity falls exactly when the peso carry does, the "diversification" is
     partly illusory and must be sized down.

Also measures the HEDGE effect (unhedged vs currency-hedged proxy): what the FX
leg adds to vol and to US-correlation.

The US-equity and carry return streams are reused from cross_correlation.py, so
the gate is consistent with the two-sleeve analysis already done.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import foreign_data as fd

OUT = os.path.join(os.path.dirname(__file__), "figures_foreign")
PPY = 12


# --- small stats helpers ---------------------------------------------------
def _sharpe(r: pd.Series, rf=0.04) -> float:
    r = r.dropna()
    if len(r) < 6:
        return np.nan
    mu, sd = r.mean() * PPY, r.std(ddof=1) * np.sqrt(PPY)
    return (mu - rf) / sd if sd > 0 else np.nan


def _tail_corr(a: pd.Series, b: pd.Series, anchor: pd.Series, q=0.20) -> float:
    """Correlation of a,b restricted to anchor's worst-q months (the tail)."""
    df = pd.concat([a, b, anchor], axis=1).dropna()
    if len(df) < 12:
        return np.nan
    x, y, z = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
    mask = z <= z.quantile(q)
    return x[mask].corr(y[mask]) if mask.sum() >= 4 else np.nan


# --- data ------------------------------------------------------------------
def load_panel(start="2005-01-01"):
    """
    Monthly return panel: the deployed US equity book, plain carry, and every
    foreign ETF proxy — all snapped to month-end. Columns keep their names; the
    two anchors are 'equity' and 'carry'. Per-pair dropna is used downstream
    (not a global dropna) so short-history funds don't truncate everything.
    """
    from cross_correlation import equity_monthly, currency_monthly
    eq = equity_monthly()                                  # deployed book
    carry, _ = currency_monthly(start=start)
    carry.index = carry.index + pd.offsets.MonthEnd(0)
    foreign = fd.download_returns(start=start)
    foreign.index = foreign.index + pd.offsets.MonthEnd(0)
    panel = pd.concat([eq.rename("equity"), carry.rename("carry"), foreign],
                      axis=1)
    return panel


# --- test 1 + 2: gate table ------------------------------------------------
FOREIGN_GROUPS = {
    "developed": list(fd.DEVELOPED_BROAD) + list(fd.DEVELOPED_COUNTRY),
    "em": list(fd.EM_BROAD) + list(fd.EM_COUNTRY),
}


def gate_table(panel: pd.DataFrame, rf=0.04) -> pd.DataFrame:
    """
    Per foreign proxy: Sharpe, corr to equity book, tail corr (equity's worst
    20% months), corr to carry, and the THRESHOLD verdict — does it earn weight?
    A proxy passes iff Sharpe_f > corr_f * Sharpe_equity (the pairwise
    diversification-earns-a-slice rule) with a non-negative Sharpe.
    """
    eq, carry = panel["equity"], panel["carry"]
    sh_eq = _sharpe(eq, rf)
    rows = {}
    for grp, names in FOREIGN_GROUPS.items():
        for name in names:
            if name not in panel:
                continue
            f = panel[name]
            pair = pd.concat([eq, f], axis=1).dropna()
            if len(pair) < 12:
                continue
            corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            sh_f = _sharpe(f, rf)
            passes = (sh_f > corr * sh_eq) and (sh_f > 0)
            rows[name] = {
                "group": grp,
                "sharpe": sh_f,
                "corr_equity": corr,
                "tail_corr_eq": _tail_corr(f, eq, eq),   # f vs eq in eq's tail
                "corr_carry": pd.concat([f, carry], axis=1).dropna()
                                .pipe(lambda d: d.iloc[:, 0].corr(d.iloc[:, 1])),
                "n": len(pair),
                "threshold": corr * sh_eq,
                "earns_weight": passes,
            }
    tab = pd.DataFrame(rows).T
    return tab.sort_values("corr_equity")


# --- test 2b: 3-asset Markowitz --------------------------------------------
def markowitz_add(panel: pd.DataFrame, foreign_name: str, rf=0.04):
    """
    Long-only max-Sharpe over {equity, carry, <foreign_name>} using raw
    annualized mean/cov (same convention as cross_correlation.markowitz_sleeve).
    Returns the tangency weights and Sharpe, plus the 2-asset baseline Sharpe,
    so we can see whether adding the foreign sleeve lifts the frontier at all.
    """
    import markowitz as mk
    cols = ["equity", "carry", foreign_name]
    df = panel[cols].dropna()
    if len(df) < 18:
        return None
    mu = df.mean().values * PPY
    Sig = df.cov().values * PPY
    w = mk.max_sharpe(mu, Sig, rf=rf)

    base = panel[["equity", "carry"]].dropna()
    mu0, Sig0 = base.mean().values * PPY, base.cov().values * PPY
    w0 = mk.max_sharpe(mu0, Sig0, rf=rf)
    return {
        "weights": dict(zip(cols, w)),
        "sharpe": mk.sharpe_ratio(w, mu, Sig, rf),
        "foreign_weight": w[2],
        "base_sharpe": mk.sharpe_ratio(w0, mu0, Sig0, rf),
        "n": len(df),
    }


# --- test 3: EM shared tail ------------------------------------------------
def em_shared_tail(panel: pd.DataFrame, rf=0.04) -> pd.DataFrame:
    """
    For each EM carry currency that has a country equity ETF: correlation of
    that country's equity to the carry book overall AND in carry's worst 20%
    months. High tail correlation => EM equity co-crashes with EM carry, so it
    is a weaker diversifier than the calm-period number suggests.
    """
    carry = panel["carry"]
    rows = {}
    for ccy, name in fd.EM_CARRY_ETF.items():
        if name not in panel:
            continue
        f = panel[name]
        pair = pd.concat([f, carry], axis=1).dropna()
        if len(pair) < 12:
            continue
        rows[f"{ccy}->{name}"] = {
            "corr_carry": pair.iloc[:, 0].corr(pair.iloc[:, 1]),
            "tail_corr_carry": _tail_corr(f, carry, carry),
            "equity_sharpe": _sharpe(f, rf),
            "n": len(pair),
        }
    return pd.DataFrame(rows).T


# --- hedge effect ----------------------------------------------------------
def hedge_effect(panel: pd.DataFrame) -> pd.DataFrame:
    """Unhedged vs currency-hedged proxy: vol and correlation-to-US, on the
    overlapping window. What hedging removes = the FX leg."""
    us = panel["equity"]
    rows = {}
    for unh, hed in fd.HEDGE_PAIRS:
        if unh not in panel or hed not in panel:
            continue
        df = pd.concat([panel[unh], panel[hed], us], axis=1).dropna()
        if len(df) < 12:
            continue
        u, h, m = df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]
        rows[unh] = {
            "vol_unhedged": u.std() * np.sqrt(PPY),
            "vol_hedged": h.std() * np.sqrt(PPY),
            "corrUS_unhedged": u.corr(m),
            "corrUS_hedged": h.corr(m),
            "n": len(df),
        }
    return pd.DataFrame(rows).T


# --- figure ----------------------------------------------------------------
def make_figure(panel: pd.DataFrame, rf=0.04):
    os.makedirs(OUT, exist_ok=True)
    tab = gate_table(panel, rf)
    em = em_shared_tail(panel, rf)
    sh_eq = _sharpe(panel["equity"], rf)
    sh_ca = _sharpe(panel["carry"], rf)

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    span = f"{panel.dropna(how='all').index[0]:%Y-%m}..{panel.dropna(how='all').index[-1]:%Y-%m}"
    fig.suptitle(f"Foreign-equity diversification gate (Tier 1, ETF proxies, {span})",
                 fontsize=15, fontweight="bold")

    # (1) corr to US: full vs tail, colored by group
    a = ax[0, 0]
    colors = {"developed": "C0", "em": "C1"}
    y = np.arange(len(tab))
    a.barh(y, tab["corr_equity"].astype(float),
           color=[colors[g] for g in tab["group"]], alpha=.85, label="full corr")
    a.scatter(tab["tail_corr_eq"].astype(float), y, color="k", zorder=5,
              s=30, label="tail corr (eq worst 20%)")
    a.set_yticks(y); a.set_yticklabels(tab.index, fontsize=8)
    a.axvline(0, color="k", lw=.8)
    a.set_title("Correlation to US equity book\n(blue=developed, orange=EM)")
    a.legend(fontsize=8); a.grid(alpha=.3, axis="x")

    # (2) the threshold gate: Sharpe vs corr*Sharpe_eq
    a = ax[0, 1]
    x = tab["corr_equity"].astype(float).values
    shf = tab["sharpe"].astype(float).values
    a.scatter(x, shf, c=[colors[g] for g in tab["group"]], s=45, zorder=5)
    for xi, yi, nm in zip(x, shf, tab.index):
        a.annotate(nm, (xi, yi), fontsize=6, alpha=.7)
    xs = np.linspace(min(0, x.min()), 1, 50)
    a.plot(xs, xs * sh_eq, "r--", lw=1, label=f"threshold = corr x {sh_eq:.2f}")
    a.fill_between(xs, xs * sh_eq, 3, color="green", alpha=.05)
    a.set_xlabel("correlation to US equity"); a.set_ylabel("proxy Sharpe")
    a.set_title("Diversification threshold\n(above line = earns weight)")
    a.legend(fontsize=8); a.grid(alpha=.3)

    # (3) EM shared-tail with carry
    a = ax[1, 0]
    if len(em):
        yy = np.arange(len(em))
        a.barh(yy - .2, em["corr_carry"].astype(float), height=.4,
               color="C1", alpha=.7, label="corr to carry (full)")
        a.barh(yy + .2, em["tail_corr_carry"].astype(float), height=.4,
               color="darkred", alpha=.8, label="corr in carry's worst 20%")
        a.set_yticks(yy); a.set_yticklabels(em.index, fontsize=8)
        a.axvline(0, color="k", lw=.8)
    a.set_title("EM equity vs EM carry: shared-tail check")
    a.legend(fontsize=8); a.grid(alpha=.3, axis="x")

    # (4) verdict text
    a = ax[1, 1]; a.axis("off")
    dev = tab[tab["group"] == "developed"]
    emr = tab[tab["group"] == "em"]
    n_dev_pass = int(dev["earns_weight"].sum())
    n_em_pass = int(emr["earns_weight"].sum())
    em_broad = "EM" if "EM" in panel.columns else None
    mk_em = markowitz_add(panel, em_broad, rf) if em_broad else None
    txt = (
        f"ANCHORS   equity Sharpe {sh_eq:.2f}   carry Sharpe {sh_ca:.2f}\n\n"
        f"DEVELOPED (expect redundant, ~0.8 corr to US):\n"
        f"  median corr to US : {dev['corr_equity'].astype(float).median():.2f}\n"
        f"  proxies clearing threshold: {n_dev_pass}/{len(dev)}\n\n"
        f"EM (the diversifier candidate):\n"
        f"  median corr to US : {emr['corr_equity'].astype(float).median():.2f}\n"
        f"  proxies clearing threshold: {n_em_pass}/{len(emr)}\n"
    )
    if mk_em:
        txt += (f"\n3-ASSET MARKOWITZ (equity, carry, EM broad):\n"
                f"  EM weight    : {mk_em['foreign_weight']*100:.0f}%\n"
                f"  Sharpe       : {mk_em['base_sharpe']:.2f} -> {mk_em['sharpe']:.2f}\n")
    if len(em):
        worst = em["tail_corr_carry"].astype(float).max()
        txt += (f"\nEM SHARED-TAIL w/ carry:\n"
                f"  max tail corr: {worst:.2f}  "
                f"({'CO-CRASH RISK' if worst > 0.5 else 'mostly independent'})\n")
    txt += ("\nREAD: value concentrates in EM. Developed clears\n"
            "the bar rarely (too US-like). Size EM by the shared-\n"
            "tail number, not the calm correlation.")
    a.text(0.02, 0.98, txt, va="top", family="monospace", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "foreign_gate.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


# --- CLI -------------------------------------------------------------------
if __name__ == "__main__":
    panel = load_panel()
    print("\n=== GATE TABLE (does each proxy earn weight?) ===")
    tab = gate_table(panel)
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(tab.round(3))

    print("\n=== EM SHARED-TAIL WITH CARRY ===")
    print(em_shared_tail(panel).round(3))

    print("\n=== HEDGE EFFECT (unhedged vs hedged) ===")
    print(hedge_effect(panel).round(3))

    if "EM" in panel.columns:
        print("\n=== 3-ASSET MARKOWITZ (equity, carry, EM broad) ===")
        print(markowitz_add(panel, "EM"))

    path = make_figure(panel)
    print(f"\nSaved gate figure -> {path}")
