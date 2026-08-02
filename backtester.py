"""
backtester.py — walk-forward backtest of the screen->allocate system.

WHAT IT DOES
------------
Simulates running the strategy through history with NO look-ahead: at each
rebalance date it uses only data available up to that date to score stocks and
set weights, then measures the realized return until the next rebalance. Rolls
forward and compounds into an equity curve, then reports CAGR / Sharpe / max
drawdown / turnover vs a benchmark.

POINT-IN-TIME HONESTY
---------------------
Only PRICE-DERIVED signals are backtested (momentum, correlation-network,
regime, ML) because they can be correctly reconstructed from past prices.
Fundamentals (value/quality) and news sentiment are EXCLUDED here: yfinance only
serves *current* snapshots, so using them historically would be look-ahead bias.
To include them you need a point-in-time fundamentals/news database.

DESIGN
------
Modular: a strategy is two functions —
    score_fn(window_prices)        -> pd.Series  (score per ticker; higher=better)
    weight_fn(window_prices, picks)-> pd.Series  (weights summing to 1)
This makes strategies swappable and testable in isolation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from utils import zscore

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# Core walk-forward engine
# ----------------------------------------------------------------------
def walk_forward(prices: pd.DataFrame, score_fn, weight_fn, *,
                 top_n=8, lookback=504, rebalance=21, cost_bps=10.0,
                 select_fn=None, adv=None, capital=1_000_000.0):
    """
    Returns a dict with the daily equity curve, the weights history, the
    per-rebalance turnover, and the daily portfolio returns.

    prices     : daily adjusted-close panel (dates x tickers)
    lookback   : trailing window (days) handed to the strategy each rebalance
    rebalance  : hold period in trading days between rebalances (21 ~ monthly)
    cost_bps   : flat one-way cost (bps) applied to turnover when `adv` is None
    adv        : optional pd.Series of dollar ADV per ticker. When provided, the
                 realistic per-name spread+impact model (costs.py) is used
                 instead of the flat cost_bps, scaled by `capital`.
    capital    : assumed portfolio NAV in dollars (sets market-impact scale)
    """
    prices = prices.sort_index()
    daily_ret = prices.pct_change(fill_method=None)
    dates = prices.index

    # rebalance dates: after the warmup, every `rebalance` days
    reb_idx = list(range(lookback, len(dates) - 1, rebalance))
    if not reb_idx:
        raise ValueError("Not enough history for one rebalance; "
                         "need > lookback + 1 rows.")

    W = pd.DataFrame(0.0, index=[dates[i] for i in reb_idx],
                     columns=prices.columns)
    for i in reb_idx:
        t = dates[i]
        window = prices.iloc[max(0, i - lookback): i + 1]   # up to & incl. t
        scores = score_fn(window)
        if scores is None or scores.dropna().empty:
            continue
        picks = (select_fn(scores) if select_fn
                 else list(scores.sort_values(ascending=False).head(top_n).index))
        if not picks:
            continue
        w = weight_fn(window, picks)
        W.loc[t, w.index] = w.values

    # forward-fill target weights to daily; yesterday's weights earn today's ret
    W_daily = W.reindex(dates).ffill().fillna(0.0)
    port_ret = (W_daily.shift(1) * daily_ret).sum(axis=1)

    # transaction costs charged on rebalance turnover
    turnover = W.diff().abs().sum(axis=1)
    turnover.iloc[0] = W.iloc[0].abs().sum()      # initial buy-in
    if adv is not None:
        from costs import rebalance_cost
        dW = W.diff()
        dW.iloc[0] = W.iloc[0]                      # initial buy-in is all change
        cost = pd.Series({t: rebalance_cost(dW.loc[t], adv, capital)
                          for t in W.index})
    else:
        cost = turnover * (cost_bps / 1e4)
    port_ret.loc[cost.index] -= cost.values

    port_ret = port_ret.iloc[lookback:].fillna(0.0)
    equity = (1 + port_ret).cumprod()
    return {"equity": equity, "returns": port_ret, "weights": W,
            "turnover": turnover}


def vol_target(returns: pd.Series, target=0.15, lookback=21, max_lev=1.0) -> pd.Series:
    """Volatility-targeting overlay: scale each day's exposure toward a constant
    annualized volatility, using only PAST realized vol (shifted, no look-ahead),
    and never lever above `max_lev` (1.0 = long-only, de-risk into cash only).
    Cuts drawdowns sharply by pulling risk down in turbulent periods."""
    realized = returns.rolling(lookback).std() * np.sqrt(TRADING_DAYS)
    scale = (target / realized).clip(upper=max_lev).shift(1).fillna(max_lev)
    return returns * scale


def benchmark_equal_weight(prices: pd.DataFrame, start_from=504):
    """Buy-and-hold equal weight of the whole universe (the thing to beat)."""
    daily_ret = prices.pct_change(fill_method=None)
    n = prices.shape[1]
    bench_ret = daily_ret.mean(axis=1).iloc[start_from:].fillna(0.0)
    return (1 + bench_ret).cumprod(), bench_ret


# ----------------------------------------------------------------------
# Performance metrics
# ----------------------------------------------------------------------
def performance(returns: pd.Series, rf=0.04) -> dict:
    r = returns.dropna()
    if len(r) < 2:
        return {k: float("nan") for k in
                ("total_return", "cagr", "vol", "sharpe", "max_dd", "n_days")}
    equity = (1 + r).cumprod()
    n = len(r)
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (TRADING_DAYS / n) - 1
    vol = r.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sharpe = (cagr - rf) / vol if vol > 0 else float("nan")
    dd = (equity / equity.cummax() - 1).min()
    return {"total_return": total, "cagr": cagr, "vol": vol,
            "sharpe": sharpe, "max_dd": dd, "n_days": n}


def summarize(name, returns, rf=0.04):
    m = performance(returns, rf)
    print(f"{name:22s} | CAGR {m['cagr']*100:6.2f}%  "
          f"Vol {m['vol']*100:5.2f}%  Sharpe {m['sharpe']:5.2f}  "
          f"MaxDD {m['max_dd']*100:6.2f}%  Total {m['total_return']*100:7.2f}%")
    return m


# ----------------------------------------------------------------------
# Built-in strategies (price-only, point-in-time safe)
# ----------------------------------------------------------------------
def weight_max_sharpe(window, picks, rf=0.04, cap=0.25, lookback=504):
    """Capped max-Sharpe weights on the trailing covariance of the picks."""
    from data import returns_stats
    from scipy.optimize import minimize
    from markowitz import sharpe_ratio
    pp = window[picks].tail(lookback)
    mu, Sig, tickers = returns_stats(pp)
    n = len(tickers)
    res = minimize(lambda w: -sharpe_ratio(w, mu, Sig, rf),
                   np.repeat(1 / n, n), method="SLSQP",
                   bounds=[(0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return pd.Series(res.x, index=tickers)


def weight_equal(window, picks):
    return pd.Series(1.0 / len(picks), index=picks)


def weight_min_variance(window, picks, cap=0.15, lookback=252, min_obs=0.5):
    """Capped minimum-variance weights on the trailing covariance of the picks.
    The min-var portfolio is the trustworthy left-tip of the frontier — unlike
    max-Sharpe it doesn't depend on fragile expected-return estimates.

    Robust to gappy point-in-time histories: names are kept if they have at least
    `min_obs` fraction of the window (so a newly-added or soon-delisted name isn't
    dropped for a single gap), covariance is computed pairwise, and if too few
    names qualify we fall back to equal weight rather than collapsing the book."""
    from scipy.optimize import minimize
    win = window[picks].tail(lookback)
    good = [c for c in picks if win[c].notna().mean() >= min_obs]
    if len(good) < 3:                    # too few to optimize -> equal weight
        valid = [c for c in picks if pd.notna(window[c].iloc[-1])] or list(picks)
        return pd.Series(1.0 / len(valid), index=valid)
    rets = win[good].pct_change(fill_method=None)
    Sig = (rets.cov().values) * TRADING_DAYS
    n = len(good)
    cap_eff = max(cap, 1.0 / n)          # keep the cap feasible for few names
    res = minimize(lambda w: w @ Sig @ w, np.repeat(1 / n, n), method="SLSQP",
                   bounds=[(0, cap_eff)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return pd.Series(res.x, index=good)


def score_reversal(window, lookback=21):
    """Short-term reversal: -(price / trailing-mean - 1). Higher = more oversold
    (the strongest honest factor in our IC study)."""
    rev = -(window / window.rolling(lookback).mean() - 1).iloc[-1]
    return zscore(rev.dropna())


def score_momentum(window):
    from factors import momentum_scores
    return zscore(momentum_scores(window))


def score_combined(window, use_ml=True):
    """Regime-weighted blend of momentum + network diversifier (+ ML)."""
    from factors import momentum_scores
    from network_model import network_scores
    from regime import detect_regime, regime_factor_weights

    mom = zscore(momentum_scores(window))
    net = network_scores(window).reindex(window.columns).fillna(0.0)
    proxy = np.log(window.mean(axis=1)).diff().dropna()   # equal-weight index
    reg = detect_regime(returns=proxy)
    rw = regime_factor_weights(reg)

    score = rw["momentum"] * mom + rw["network"] * net
    if use_ml:
        from ml_rank import ml_rank_scores
        ml = ml_rank_scores(window).reindex(window.columns).fillna(0.0)
        score = 0.75 * score + 0.25 * ml
    return zscore(score)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(use_ml=False):
    from data import download_prices
    from screener import DEFAULT_UNIVERSE

    prices = download_prices(DEFAULT_UNIVERSE)
    print(f"Backtesting {prices.shape[1]} stocks, "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}  "
          f"(ML={'on' if use_ml else 'off'})\n")

    res = walk_forward(
        prices,
        score_fn=lambda w: score_combined(w, use_ml=use_ml),
        weight_fn=lambda w, p: weight_max_sharpe(w, p),
        top_n=8, lookback=504, rebalance=21, cost_bps=10.0,
    )
    _, bench_ret = benchmark_equal_weight(prices, start_from=504)
    bench_ret = bench_ret.reindex(res["returns"].index).fillna(0.0)

    print("Strategy vs benchmark (after 10bps costs):")
    summarize("Strategy (combined)", res["returns"])
    summarize("Equal-weight universe", bench_ret)
    avg_turn = res["turnover"].mean()
    print(f"\nAvg turnover per rebalance: {avg_turn*100:.1f}%  "
          f"(~{res['turnover'].shape[0]} rebalances)")
    print("\nNOTE: price-based signals only (point-in-time safe). Value/quality/"
          "sentiment excluded — they need a PIT database to backtest honestly.")
    return res


if __name__ == "__main__":
    import sys
    main(use_ml="--ml" in sys.argv)
