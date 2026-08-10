"""
commodity_trend.py — the trend (time-series momentum) backtest on REAL futures.

Build-order step 3: confirm on real IBKR futures (not the ETF proxy) that a proper,
vol-targeted trend book earns a positive return AND keeps the crisis-alpha property
the Tier-1 gate found (rises in the equity book's worst months).

Construction (managed-futures standard, all no-lookahead):
  1. trend score per market = average SIGN of trailing 3/6/12-month returns,
     lagged one month (uses only data through t-1).
  2. per-market INVERSE-VOL sizing (scale by 1/trailing-vol) so no single market
     dominates the risk; risk-normalized to gross |w| = 1 each month.
  3. portfolio VOL-TARGET overlay (same idea as the equity book): scale the whole
     book by target/trailing-book-vol, capped — a timing overlay on gross exposure.
  4. realize NEXT month's return net of turnover cost.

Reported against: the equal-weight-sign book (gate style) and the equity book
(the crisis-alpha check). Uses the cached real panel from commodity_futures.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import MONTH_END

OUT = os.path.join(os.path.dirname(__file__), "figures_commodity")
PPY = 12


# --- returns ---------------------------------------------------------------
def load_returns() -> pd.DataFrame:
    """Monthly returns of the real continuous futures panel."""
    import commodity_futures as cfut
    px = cfut.load_continuous(monthly=True)
    return px.pct_change()


# --- signal + sizing (all lagged -> no lookahead) --------------------------
def trend_score(rets: pd.DataFrame, lookbacks=(3, 6, 12)) -> pd.DataFrame:
    """Average sign of trailing L-month returns, in [-1,1], LAGGED one month."""
    score = None
    for L in lookbacks:
        mom = (1 + rets).rolling(L).apply(np.prod, raw=True) - 1
        s = np.sign(mom)
        score = s if score is None else score + s
    return (score / len(lookbacks)).shift(1)


def inv_vol(rets: pd.DataFrame, lookback=12) -> pd.DataFrame:
    """1 / trailing annualized vol per market, LAGGED (for equal-risk sizing)."""
    vol = rets.rolling(lookback).std() * np.sqrt(PPY)
    return (1.0 / vol.replace(0.0, np.nan)).shift(1)


def weights(rets, lookbacks=(3, 6, 12), vol_lookback=12) -> pd.DataFrame:
    """Signed, inverse-vol, risk-normalized weights (gross |w| = 1 each month).
    Already lagged via trend_score / inv_vol, so w_t uses only data through t-1."""
    raw = trend_score(rets, lookbacks) * inv_vol(rets, vol_lookback)
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    return raw.div(gross, axis=0)


# --- backtest --------------------------------------------------------------
def backtest(rets=None, target_vol=0.12, cap=2.0, cost_bps=3.0,
             lookbacks=(3, 6, 12), vol_lookback=12, vol_target=True) -> dict:
    """
    Vol-targeted trend book. w already lagged, so gross_t = sum_i w_{i,t}*ret_{i,t}
    is a genuine out-of-sample realization. Optional portfolio vol-target overlay.
    """
    if rets is None:
        rets = load_returns()
    w = weights(rets, lookbacks, vol_lookback)
    gross = (w * rets).sum(axis=1, min_count=1)

    # turnover cost on |Δweight|
    dw = (w - w.shift(1)).abs().sum(axis=1)
    cost = dw * cost_bps / 1e4

    if vol_target:
        book_vol = gross.rolling(12).std() * np.sqrt(PPY)
        e = (target_vol / book_vol).clip(upper=cap).shift(1)
    else:
        e = pd.Series(1.0, index=gross.index)
    net = (e * gross - cost).dropna()

    return {"net": net, "gross": gross, "exposure": e, "weights": w,
            "turnover": dw}


# --- metrics ---------------------------------------------------------------
def performance(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 6:
        return {k: np.nan for k in ("sharpe", "cagr", "vol", "maxdd", "skew",
                                    "worst", "hit", "n")}
    mu, sd = r.mean(), r.std(ddof=1)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    yrs = len(r) / PPY
    return {"sharpe": mu / sd * np.sqrt(PPY) if sd > 0 else np.nan,
            "cagr": eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else np.nan,
            "vol": sd * np.sqrt(PPY), "maxdd": dd, "skew": r.skew(),
            "worst": r.min(), "hit": (r > 0).mean(), "n": len(r)}


def crisis_alpha(trend_net: pd.Series, q=0.20) -> dict:
    """Correlation to the equity book + return in the equity book's worst q months
    (the diversification/tail-hedge check) on real data."""
    from cross_correlation import equity_monthly
    eq = equity_monthly()
    df = pd.concat([eq.rename("eq"), trend_net.rename("tr")], axis=1).dropna()
    if len(df) < 12:
        return {"n": len(df)}
    thr = df["eq"].quantile(q)
    stress = df["eq"] <= thr
    return {"corr": df["eq"].corr(df["tr"]),
            "tail_corr": df["eq"][stress].corr(df["tr"][stress]),
            "trend_mean_stress": df["tr"][stress].mean(),
            "eq_mean_stress": df["eq"][stress].mean(),
            "n": len(df), "n_stress": int(stress.sum())}


# --- figure ----------------------------------------------------------------
def make_figure(res: dict, rets: pd.DataFrame):
    os.makedirs(OUT, exist_ok=True)
    net = res["net"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    span = f"{net.index[0]:%Y-%m}..{net.index[-1]:%Y-%m}"
    fig.suptitle(f"Commodity trend book — real IBKR futures ({span}, n={len(net)})",
                 fontsize=14, fontweight="bold")

    a = ax[0, 0]
    a.plot((1 + net).cumprod(), color="C1")
    a.set_yscale("log"); a.set_title("Growth of $1 (vol-targeted, net)"); a.grid(alpha=.3)

    a = ax[0, 1]
    e = (1 + net).cumprod()
    a.fill_between(e.index, (e / e.cummax() - 1).values, 0, color="C3", alpha=.5)
    a.set_title("Drawdown"); a.grid(alpha=.3)

    a = ax[1, 0]
    try:
        from cross_correlation import equity_monthly
        eq = equity_monthly()
        df = pd.concat([eq.rename("equity"), net.rename("trend")], axis=1).dropna()
        thr = df["equity"].quantile(0.20)
        a.plot((1 + df["equity"]).cumprod(), color="C0", label="equity book")
        a.plot((1 + df["trend"]).cumprod(), color="C1", label="commodity trend")
        for t in df.index[df["equity"] <= thr]:
            a.axvspan(t - pd.offsets.MonthEnd(1), t, color="red", alpha=.08)
        a.set_yscale("log"); a.set_title("vs equity (red = equity's worst months)")
        a.legend(fontsize=9); a.grid(alpha=.3)
    except Exception as e:
        a.axis("off"); a.text(0.1, 0.5, f"equity overlay n/a: {str(e)[:40]}")

    a = ax[1, 1]; a.axis("off")
    m = performance(net)
    ca = crisis_alpha(net)
    txt = (f"VOL-TARGETED TREND BOOK (net, real futures)\n\n"
           f"  Sharpe   {m['sharpe']:.2f}\n  CAGR     {m['cagr']*100:.1f}%\n"
           f"  Vol      {m['vol']*100:.1f}%\n  MaxDD    {m['maxdd']*100:.1f}%\n"
           f"  Skew     {m['skew']:.2f}\n  Worst mo {m['worst']*100:.1f}%\n"
           f"  Hit      {m['hit']*100:.0f}%\n  Months   {m['n']}\n\n"
           f"CRISIS ALPHA vs equity book:\n")
    if ca.get("n", 0) >= 12:
        txt += (f"  corr           {ca['corr']:+.2f}\n"
                f"  tail corr      {ca['tail_corr']:+.2f}\n"
                f"  trend ret in eq's worst {ca['n_stress']} mo: "
                f"{ca['trend_mean_stress']*100:+.2f}%/mo\n"
                f"  (equity then:  {ca['eq_mean_stress']*100:+.2f}%/mo)")
    a.text(0.02, 0.98, txt, va="top", family="monospace", fontsize=10.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, "commodity_trend.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


# --- CLI -------------------------------------------------------------------
if __name__ == "__main__":
    rets = load_returns()
    print(f"Real futures panel: {rets.shape[1]} markets, "
          f"{rets.index[0]:%Y-%m}..{rets.index[-1]:%Y-%m} ({len(rets)} months)\n")

    vt = backtest(rets, vol_target=True)
    raw = backtest(rets, vol_target=False)

    print(f"{'book':<28}{'Sharpe':>8}{'CAGR':>8}{'Vol':>7}{'MaxDD':>8}"
          f"{'Skew':>7}{'n':>5}")
    for name, r in [("vol-targeted trend", vt["net"]),
                    ("trend (no vol-target)", raw["net"])]:
        m = performance(r)
        print(f"{name:<28}{m['sharpe']:>8.2f}{m['cagr']*100:>7.1f}%"
              f"{m['vol']*100:>6.1f}%{m['maxdd']*100:>7.1f}%{m['skew']:>7.2f}{m['n']:>5}")

    ca = crisis_alpha(vt["net"])
    if ca.get("n", 0) >= 12:
        print(f"\nCRISIS ALPHA (vs equity book, {ca['n']} overlapping months):")
        print(f"  corr {ca['corr']:+.2f}   tail corr {ca['tail_corr']:+.2f}")
        print(f"  trend return in equity's worst {ca['n_stress']} months: "
              f"{ca['trend_mean_stress']*100:+.2f}%/mo  "
              f"(equity then {ca['eq_mean_stress']*100:+.2f}%/mo)")

    path = make_figure(vt, rets)
    print(f"\nSaved figure -> {path}")
