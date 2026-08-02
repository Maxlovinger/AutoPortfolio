"""
data.py — fetch price history and turn it into the (mu, Sigma) inputs
Markowitz needs.

Key modeling choices explained inline, because they matter more than the
optimizer itself:
  * We use LOG returns for stability.
  * We ANNUALIZE (daily stats * 252) so numbers are interpretable.
  * Expected return from historical mean is NAIVE and noisy — good enough to
    learn the frontier, but this is the #1 thing to improve later
    (Black-Litterman, factor models, shrinkage).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252


def download_prices(tickers, start="2018-01-01", end=None) -> pd.DataFrame:
    """Adjusted close prices, one column per ticker."""
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )
    prices = raw["Close"] if "Close" in raw else raw
    prices = prices.dropna(how="all").ffill().dropna()
    return prices


def returns_stats(prices: pd.DataFrame):
    """
    Return (mu, Sigma, tickers):
        mu    = annualized expected return per asset (numpy vector)
        Sigma = annualized covariance matrix (numpy 2D)
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    mu = log_ret.mean().values * TRADING_DAYS
    Sigma = log_ret.cov().values * TRADING_DAYS
    return mu, Sigma, list(prices.columns)
