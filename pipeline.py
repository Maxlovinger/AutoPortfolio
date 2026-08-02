"""
pipeline.py — the full system, end to end.

    SCREEN (alpha)                         ALLOCATE (Markowitz)
    ┌───────────────────────────┐          ┌─────────────────────┐
    │ A factors                 │          │ mu, Sigma           │
    │ B sentiment   ─┐          │  top-N   │ max-Sharpe weights  │
    │ C network      ├─ fuse ── │ ───────► │ min-variance weights│
    │ D regime (wts) │          │ stocks   │ efficient frontier  │
    │ E ml rank     ─┘          │          └─────────────────────┘
    └───────────────────────────┘

Run:  python3 pipeline.py
This is a RESEARCH prototype. Do NOT trade it before building the walk-forward
backtest (next step) — the scores are in-sample and unvalidated.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from data import download_prices, returns_stats
from screener import run_screen, top_n
from markowitz import (
    max_sharpe, min_variance,
    portfolio_return, portfolio_vol, sharpe_ratio,
)

TOP_N = 10
RISK_FREE = 0.04
MAX_WEIGHT = 0.25   # cap any single stock to force diversification


def show(name, w, mu, Sig, tickers):
    print(f"\n=== {name} ===")
    for t, wi in sorted(zip(tickers, w), key=lambda x: -x[1]):
        if wi > 0.005:
            print(f"  {t:6s} {wi*100:6.2f}%")
    print(f"  -> return {portfolio_return(w, mu)*100:5.2f}%  "
          f"vol {portfolio_vol(w, Sig)*100:5.2f}%  "
          f"Sharpe {sharpe_ratio(w, mu, Sig, RISK_FREE):4.2f}")


def main():
    # 1) SCREEN — pick the stocks
    screen, regime = run_screen()
    picks = top_n(screen, TOP_N, one_per_community=False)
    print(f"\nTop {TOP_N} stocks by fused score: {picks}")

    # 2) ALLOCATE — Markowitz on just the winners
    prices = download_prices(picks)
    mu, Sig, tickers = returns_stats(prices)

    # capped weights so no single name dominates
    from scipy.optimize import minimize
    n = len(tickers)
    bounds = [(0.0, MAX_WEIGHT)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    w_ms = minimize(lambda w: -sharpe_ratio(w, mu, Sig, RISK_FREE),
                    np.repeat(1/n, n), method="SLSQP",
                    bounds=bounds, constraints=cons).x
    w_mv = min_variance(Sig)

    show(f"Max-Sharpe portfolio (capped {int(MAX_WEIGHT*100)}%/name)",
         w_ms, mu, Sig, tickers)
    show("Min-Variance portfolio", w_mv, mu, Sig, tickers)

    print("\nNOTE: in-sample research output. Validate with a walk-forward "
          "backtest before trusting these numbers.")


if __name__ == "__main__":
    main()
