"""
run.py — end-to-end demo of the Markowitz efficient frontier.

Pipeline:
    1. Pick a STARTER UNIVERSE (diversified ETFs across asset classes)
    2. Download prices  -> mu, Sigma
    3. Compute the efficient frontier + the two key portfolios
    4. Print weights and plot the frontier

Run:  python run.py
Output: prints tables + saves frontier.png
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from data import download_prices, returns_stats
from markowitz import (
    efficient_frontier,
    max_sharpe,
    min_variance,
    portfolio_return,
    portfolio_vol,
    sharpe_ratio,
)

# ----------------------------------------------------------------------
# 1. STARTER UNIVERSE — deliberately broad & diversified.
#    This is a LEARNING universe, not a stock-picking result.
#    Swap these out later once you have a real screening process.
# ----------------------------------------------------------------------
UNIVERSE = {
    "VTI":  "US total stock market",
    "VXUS": "International stocks",
    "QQQ":  "US tech / growth",
    "VNQ":  "Real estate (REITs)",
    "BND":  "US total bond market",
    "TLT":  "Long-term treasuries",
    "GLD":  "Gold",
}
RISK_FREE = 0.04  # ~current short T-bill yield; used for Sharpe


def show(name, w, mu, Sig, tickers):
    print(f"\n=== {name} ===")
    ret = portfolio_return(w, mu)
    vol = portfolio_vol(w, Sig)
    shp = sharpe_ratio(w, mu, Sig, RISK_FREE)
    for t, wi in sorted(zip(tickers, w), key=lambda x: -x[1]):
        if wi > 0.001:
            print(f"  {t:5s} {wi*100:6.2f}%")
    print(f"  -> Expected return: {ret*100:5.2f}%   "
          f"Volatility: {vol*100:5.2f}%   Sharpe: {shp:4.2f}")


def main():
    tickers = list(UNIVERSE)
    print(f"Downloading {len(tickers)} assets: {', '.join(tickers)}")
    prices = download_prices(tickers)
    print(f"Got {len(prices)} trading days: "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}")

    mu, Sig, tickers = returns_stats(prices)

    # 3. Key portfolios
    w_mv = min_variance(Sig)
    w_ms = max_sharpe(mu, Sig, RISK_FREE)
    show("Minimum-Variance Portfolio (lowest risk)", w_mv, mu, Sig, tickers)
    show("Maximum-Sharpe Portfolio (best risk-adjusted)", w_ms, mu, Sig, tickers)

    # 4. Frontier + plot
    vols, rets, _ = efficient_frontier(mu, Sig, n_points=60)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 6))
        # individual assets
        for i, t in enumerate(tickers):
            plt.scatter(np.sqrt(Sig[i, i]), mu[i], marker="o")
            plt.annotate(t, (np.sqrt(Sig[i, i]), mu[i]))
        plt.plot(vols, rets, "b-", lw=2, label="Efficient frontier")
        plt.scatter(portfolio_vol(w_ms, Sig), portfolio_return(w_ms, mu),
                    c="red", marker="*", s=250, label="Max Sharpe")
        plt.scatter(portfolio_vol(w_mv, Sig), portfolio_return(w_mv, mu),
                    c="green", marker="*", s=250, label="Min Variance")
        plt.xlabel("Volatility (annualized)")
        plt.ylabel("Expected return (annualized)")
        plt.title("Markowitz Efficient Frontier")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("frontier.png", dpi=120)
        print("\nSaved plot -> frontier.png")
    except Exception as e:
        print(f"\n(Plot skipped: {e})")


if __name__ == "__main__":
    main()
