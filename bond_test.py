"""
bond_test.py — two ways to add bonds, head to head, with a 2022 stress test.

Baseline : the deployed equity book, vol-target de-risking to CASH (earns rf).
Approach 1 (STRATEGIC): hold a continuous bond allocation blended with equity
           (60/40-style), rebalanced monthly.
Approach 2 (ROUTE DE-RISK): equity book, but the de-risked slice (1 - exposure)
           goes into BONDS instead of cash.

  routed_t   = overlaid_t + (1 - scale_t) * bond_t     (bonds only when de-risked)
  baseline_t = overlaid_t + (1 - scale_t) * rf_t       (cash earns risk-free)
  strategic_t= (1-w) * baseline_t + w * bond_t          (bonds always on)

The honest crux is the 2022 inflation shock, where stock-bond correlation flipped
positive (bonds fell WITH stocks) — so we report full-sample Sharpe/DD AND the
2022 return and the return in equities' worst months.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

from utils import MONTH_END
import markowitz as mk

PPY = 12
BONDS = ["SHY", "IEF", "TLT"]        # 1-3y, 7-10y, 20y+ Treasuries


def _rf_daily(index):
    """Daily risk-free from FRED 3M T-bill (DGS3MO, % annual)."""
    from fx.data import _fred_client
    s = _fred_client().get_series("DGS3MO").dropna() / 100.0
    s.index = pd.to_datetime(s.index)
    return (s.reindex(index.union(s.index)).ffill().reindex(index) / 252.0)


def load():
    import final_strategy as fs
    d = fs.build()
    raw, scale, overlaid = d["raw"], d["scale"], d["overlaid"]
    idx = overlaid.dropna().index
    px = yf.download(BONDS, start="2015-01-01", auto_adjust=True, progress=False)
    px = px["Close"] if "Close" in px else px
    bond_ret = px.reindex(idx.union(px.index)).ffill().reindex(idx).pct_change()
    rf = _rf_daily(idx)
    return (overlaid.reindex(idx), scale.reindex(idx),
            bond_ret, rf.reindex(idx))


# --- metrics ---------------------------------------------------------------
def stats(daily: pd.Series) -> dict:
    r = daily.dropna()
    m = (1 + r).resample(MONTH_END).prod() - 1        # monthly for skew/DD read
    mu, sd = r.mean(), r.std(ddof=1)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    ret2022 = (1 + r.loc["2022"]).prod() - 1 if "2022" in r.index.strftime("%Y") else np.nan
    return {"sharpe": mu / sd * np.sqrt(252) if sd > 0 else np.nan,
            "cagr": eq.iloc[-1] ** (252 / len(r)) - 1 if eq.iloc[-1] > 0 else np.nan,
            "vol": sd * np.sqrt(252), "maxdd": dd,
            "ret2022": ret2022, "monthly": m}


def worst_months_return(port_daily, equity_daily, q=0.20):
    """Mean monthly return of `port` in equities' worst-q months (the hedge test)."""
    p = (1 + port_daily).resample(MONTH_END).prod() - 1
    e = (1 + equity_daily).resample(MONTH_END).prod() - 1
    df = pd.concat([e.rename("e"), p.rename("p")], axis=1).dropna()
    stress = df["e"] <= df["e"].quantile(q)
    return df["p"][stress].mean(), df["e"][stress].mean()


def run():
    overlaid, scale, bond_ret, rf = load()
    derisk = 1 - scale
    baseline = overlaid + derisk * rf                 # de-risk -> cash (rf)

    span = f"{overlaid.index[0]:%Y-%m}..{overlaid.index[-1]:%Y-%m}"
    print(f"Equity book + bonds, daily {span}\n")
    b0 = stats(baseline)
    print(f"{'portfolio':<26}{'Sharpe':>8}{'CAGR':>7}{'Vol':>7}{'MaxDD':>8}"
          f"{'2022':>8}{'worstMo':>9}")

    def _row(name, daily):
        s = stats(daily)
        wm, _ = worst_months_return(daily, baseline)
        print(f"{name:<26}{s['sharpe']:>8.2f}{s['cagr']*100:>6.1f}%{s['vol']*100:>6.1f}%"
              f"{s['maxdd']*100:>7.1f}%{s['ret2022']*100:>7.1f}%{wm*100:>8.2f}%")
        return s

    _row("baseline (de-risk->cash)", baseline)

    # Approach 2 — route de-risk into each bond
    print("  --- Approach 2: ROUTE de-risk into bonds ---")
    for b in BONDS:
        _row(f"  route->{b}", overlaid + derisk * bond_ret[b])

    # Approach 1 — strategic continuous blend (fixed + Markowitz-optimal weight)
    print("  --- Approach 1: STRATEGIC continuous blend ---")
    for b in ["IEF", "TLT"]:
        # markowitz-optimal long-only weight on (baseline, bond) monthly
        m = pd.concat([( (1+baseline).resample(MONTH_END).prod()-1 ).rename("eq"),
                       ( (1+bond_ret[b]).resample(MONTH_END).prod()-1 ).rename("bd")],
                      axis=1).dropna()
        mu, Sig = m.mean().values * PPY, m.cov().values * PPY
        w = mk.max_sharpe(mu, Sig, rf=0.0)
        wb = w[1]
        for label, wbond in [(f"  20% {b}", 0.20), (f"  40% {b}", 0.40),
                             (f"  MV-opt {wb*100:.0f}% {b}", wb)]:
            _row(label, (1 - wbond) * baseline + wbond * bond_ret[b])

    print("\nRead: compare each row's Sharpe/MaxDD to baseline, and check the 2022 "
          "column (inflation shock) and worstMo (return in equity's worst months).")


if __name__ == "__main__":
    run()
