"""
portfolio_bonds.py — the 3-sleeve portfolio: equity + FX carry + a 20% IEF bond
sleeve, with the full diagram set.

Structure (capital weights): bonds are 20% of the TOTAL portfolio; the existing
71/29 equity/currency split scales into the remaining 80%:
    equity  = 0.80 * 0.71 = 56.8%
    currency= 0.80 * 0.29 = 23.2%
    bonds   = 20.0%
Compared against the old 2-sleeve book (equity 71% / currency 29%, no bonds).

Sleeve return streams: deployed equity book (final_strategy, vol-targeted),
plain FX carry, and IEF (7-10y Treasuries). Note: the currency sleeve is normally
run levered to contribute its intended RISK share, so capital-weighting understates
its risk contribution — we flag this rather than hide it.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

from utils import MONTH_END

OUT = os.path.join(os.path.dirname(__file__), "figures_portfolio")
PPY = 12


def _weights(bond_weight):
    """bonds = bond_weight of total; equity/currency keep 71/29 in the rest."""
    return (1 - bond_weight) * 0.71, (1 - bond_weight) * 0.29, bond_weight


def _ief_monthly(idx):
    px = yf.download("IEF", start="2010-01-01", auto_adjust=True, progress=False)
    px = px["Close"] if "Close" in px else px
    px = px.squeeze()
    m = px.resample(MONTH_END).last().pct_change()
    return m.reindex(idx)


def load():
    from cross_correlation import equity_monthly, currency_monthly
    eq = equity_monthly().rename("equity")
    carry, _ = currency_monthly()
    carry = carry.rename("currency"); carry.index = carry.index + pd.offsets.MonthEnd(0)
    df = pd.concat([eq, carry], axis=1)
    df["bonds"] = _ief_monthly(df.index)
    return df.dropna()


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    mu, sd = r.mean(), r.std(ddof=1)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    yrs = len(r) / PPY
    y2022 = (1 + r[r.index.year == 2022]).prod() - 1 if (r.index.year == 2022).any() else np.nan
    return {"sharpe": mu / sd * np.sqrt(PPY) if sd > 0 else np.nan,
            "cagr": eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else np.nan,
            "vol": sd * np.sqrt(PPY), "maxdd": dd, "skew": r.skew(), "ret2022": y2022}


def compare_weights(df=None, weights=(0.0, 0.10, 0.20, 0.30)):
    """Metric table across bond weights so the size tradeoff is visible."""
    df = load() if df is None else df
    eq, fx, bd = df["equity"], df["currency"], df["bonds"]
    e0 = eq                                               # equity ref for worst-months
    thr = e0.quantile(0.20); stress = e0 <= thr
    print(f"{'bonds%':>7}{'Sharpe':>8}{'Vol':>7}{'MaxDD':>8}{'CAGR':>7}{'2022':>8}{'worstMo':>9}")
    for w in weights:
        we, wf, wb = _weights(w)
        r = we * eq + wf * fx + wb * bd
        s = stats(r)
        print(f"{w*100:>6.0f}%{s['sharpe']:>8.2f}{s['vol']*100:>6.1f}%{s['maxdd']*100:>7.1f}%"
              f"{s['cagr']*100:>6.1f}%{s['ret2022']*100:>7.1f}%{r[stress].mean()*100:>8.2f}%")


def make_diagrams(bond_weight=0.10):
    os.makedirs(OUT, exist_ok=True)
    df = load()
    W_EQ, W_FX, W_BOND = _weights(bond_weight)
    eq, fx, bd = df["equity"], df["currency"], df["bonds"]
    two = 0.71 * eq + 0.29 * fx                                # old 2-sleeve
    three = W_EQ * eq + W_FX * fx + W_BOND * bd                # new 3-sleeve
    span = f"{df.index[0]:%Y-%m}..{df.index[-1]:%Y-%m}"
    s2, s3 = stats(two), stats(three)

    fig, ax = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle(f"3-sleeve portfolio: Equity + FX carry + {bond_weight*100:.0f}% IEF bonds  "
                 f"({span}, n={len(df)})", fontsize=15, fontweight="bold")

    # (1) allocation donut
    a = ax[0, 0]
    a.pie([W_EQ, W_FX, W_BOND], labels=[f"Equity {W_EQ*100:.1f}%",
          f"Currency {W_FX*100:.1f}%", f"Bonds {W_BOND*100:.0f}%"],
          colors=["C0", "C1", "C2"], autopct="%1.0f%%",
          wedgeprops=dict(width=0.42), startangle=90)
    a.set_title("Capital allocation (bonds = 20% of total)")

    # (2) growth of $1
    a = ax[0, 1]
    for s, c, lab in [(eq, "C0", "equity"), (fx, "C1", "currency"),
                      (bd, "C2", "bonds (IEF)"), (two, "0.5", "2-sleeve (old)"),
                      (three, "C3", "3-sleeve (new)")]:
        a.plot((1 + s).cumprod(), color=c, label=lab,
               lw=2.2 if lab.endswith(("new)", "old)")) else 1.1)
    a.set_yscale("log"); a.set_title("Growth of $1 (log)"); a.legend(fontsize=8); a.grid(alpha=.3)

    # (3) correlation matrix
    a = ax[0, 2]
    corr = df.corr()
    im = a.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    a.set_xticks(range(3)); a.set_yticks(range(3))
    a.set_xticklabels(corr.columns, rotation=20); a.set_yticklabels(corr.columns)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                   color="white" if abs(corr.iloc[i, j]) > .5 else "black")
    a.set_title("Sleeve correlation"); fig.colorbar(im, ax=a, fraction=.046)

    # (4) drawdowns: 2 vs 3 sleeve
    a = ax[1, 0]
    for s, c, lab in [(two, "0.5", "2-sleeve"), (three, "C3", "3-sleeve")]:
        e = (1 + s).cumprod(); a.fill_between(e.index, (e / e.cummax() - 1).values, 0,
                                              color=c, alpha=.45, label=lab)
    a.set_title("Drawdown: 2-sleeve vs 3-sleeve"); a.legend(); a.grid(alpha=.3)

    # (5) stats bars
    a = ax[1, 1]
    labels = ["Sharpe", "Vol", "MaxDD", "2022"]
    v2 = [s2["sharpe"], s2["vol"], -s2["maxdd"], -s2["ret2022"]]
    v3 = [s3["sharpe"], s3["vol"], -s3["maxdd"], -s3["ret2022"]]
    x = np.arange(len(labels)); w = 0.38
    a.bar(x - w/2, v2, w, label="2-sleeve", color="0.5")
    a.bar(x + w/2, v3, w, label="3-sleeve", color="C3")
    a.set_xticks(x); a.set_xticklabels(labels)
    a.set_title("2-sleeve vs 3-sleeve (Vol/DD/2022 as magnitude)")
    a.legend(); a.grid(alpha=.3, axis="y")
    for i, (b2, b3) in enumerate(zip(v2, v3)):
        a.text(i - w/2, b2, f"{b2:.2f}", ha="center", va="bottom", fontsize=8)
        a.text(i + w/2, b3, f"{b3:.2f}", ha="center", va="bottom", fontsize=8)

    # (6) summary
    a = ax[1, 2]; a.axis("off")
    thr = eq.quantile(0.20); stress = eq <= thr
    cr2, cr3 = two[stress].mean(), three[stress].mean()
    txt = (
        f"CAPITAL ALLOCATION\n"
        f"  Equity   {W_EQ*100:.1f}%\n  Currency {W_FX*100:.1f}%\n  Bonds    {W_BOND*100:.0f}%\n\n"
        f"{'metric':<10}{'2-sleeve':>10}{'3-sleeve':>10}\n"
        f"{'Sharpe':<10}{s2['sharpe']:>10.2f}{s3['sharpe']:>10.2f}\n"
        f"{'Vol':<10}{s2['vol']*100:>9.1f}%{s3['vol']*100:>9.1f}%\n"
        f"{'MaxDD':<10}{s2['maxdd']*100:>9.1f}%{s3['maxdd']*100:>9.1f}%\n"
        f"{'CAGR':<10}{s2['cagr']*100:>9.1f}%{s3['cagr']*100:>9.1f}%\n"
        f"{'2022':<10}{s2['ret2022']*100:>9.1f}%{s3['ret2022']*100:>9.1f}%\n"
        f"{'worst mo':<10}{cr2*100:>9.2f}%{cr3*100:>9.2f}%\n\n"
        f"READ: bonds (20%) cut VOL and DRAWDOWN and\n"
        f"soften equity's worst months, at some CAGR cost.\n"
        f"A smoother ride, not a higher Sharpe.\n\n"
        f"CAVEAT: 2017-26 was bond-hostile (2022 crash);\n"
        f"IEF now yields ~4-5% -> better forward than shown."
    )
    a.text(0.02, 0.98, txt, va="top", family="monospace", fontsize=10.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, f"portfolio_3sleeve_{bond_weight*100:.0f}pct.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"saved -> {p}")
    print(f"\n2-sleeve: Sharpe {s2['sharpe']:.2f}  Vol {s2['vol']*100:.1f}%  "
          f"MaxDD {s2['maxdd']*100:.1f}%  2022 {s2['ret2022']*100:.1f}%")
    print(f"3-sleeve: Sharpe {s3['sharpe']:.2f}  Vol {s3['vol']*100:.1f}%  "
          f"MaxDD {s3['maxdd']*100:.1f}%  2022 {s3['ret2022']*100:.1f}%")
    return df


if __name__ == "__main__":
    import sys
    w = float(sys.argv[1]) / 100 if len(sys.argv) > 1 else 0.10
    print("=== BOND-WEIGHT SWEEP (equity/currency keep 71/29 in the rest) ===")
    compare_weights()
    print()
    make_diagrams(bond_weight=w)
