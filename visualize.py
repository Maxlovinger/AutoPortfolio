"""
visualize.py — see what every model is doing.

Design: the plot_* functions are PURE — they take already-computed data and save
a PNG. That makes them testable offline with synthetic data. `make_all()` is the
orchestrator that pulls live data, runs the models once, and renders everything
into ./figures/.

Run:  python3 visualize.py          # generates all figures into ./figures/
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")               # headless / file output
import matplotlib.pyplot as plt

FIGDIR = "figures"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# Prices & correlation
# ----------------------------------------------------------------------
def plot_price_history(prices: pd.DataFrame, path: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    norm = prices / prices.iloc[0] * 100      # rebased to 100
    norm.plot(ax=ax, lw=1.2)
    ax.set_title("Price history (rebased to 100)")
    ax.set_ylabel("Growth of 100")
    ax.legend(ncol=4, fontsize=7)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_correlation_heatmap(prices: pd.DataFrame, path: str):
    rets = np.log(prices / prices.shift(1)).dropna()
    corr = rets.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title("Return correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, path)


# ----------------------------------------------------------------------
# (A) Factors
# ----------------------------------------------------------------------
def plot_factor_scores(screen: pd.DataFrame, path: str, cols=None):
    cols = [c for c in (cols or ["value", "quality", "momentum", "value_x_mom"])
            if c in screen.columns]
    data = screen[cols]
    fig, ax = plt.subplots(figsize=(12, 6))
    data.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title("(A) Factor scores per stock (z-scored)")
    ax.set_ylabel("z-score")
    ax.axhline(0, color="k", lw=0.7)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


# ----------------------------------------------------------------------
# (B) Sentiment
# ----------------------------------------------------------------------
def plot_sentiment(sent: pd.Series, path: str):
    s = sent.sort_values()
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in s.values]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(s.index.astype(str), s.values, color=colors)
    ax.set_title("(B) News-sentiment score per stock (z-scored)")
    ax.set_ylabel("sentiment z-score")
    ax.axhline(0, color="k", lw=0.7)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


# ----------------------------------------------------------------------
# (C) Network
# ----------------------------------------------------------------------
def plot_network(prices: pd.DataFrame, path: str, threshold: float = 0.4):
    import networkx as nx
    from network_model import build_correlation_graph, community_labels

    G = build_correlation_graph(prices, threshold)
    comms = community_labels(prices, threshold)
    fig, ax = plt.subplots(figsize=(9, 8))

    if G.number_of_edges() == 0:
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, weight="weight")

    node_colors = [comms.get(n, 0) for n in G.nodes()]
    try:
        cent = nx.degree_centrality(G)
        sizes = [300 + 2500 * cent.get(n, 0) for n in G.nodes()]
    except Exception:
        sizes = 500

    if G.number_of_edges() > 0:
        weights = [G[u][v]["weight"] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax, width=weights, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                           node_color=node_colors, cmap="tab10", alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    ax.set_title("(C) Correlation network — color=cluster, size=centrality")
    ax.axis("off")
    return _save(fig, path)


# ----------------------------------------------------------------------
# (D) Regime
# ----------------------------------------------------------------------
def plot_regime(returns: pd.Series, p_stress: pd.Series, path: str):
    price = (1 + returns).cumprod()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   height_ratios=[2, 1])
    ax1.plot(price.index, price.values, color="navy", lw=1)
    ax1.set_title("(D) Market with stress-regime shading")
    ax1.set_ylabel("cumulative")
    if len(p_stress):
        ps = p_stress.reindex(price.index).fillna(0.0)
        ax1.fill_between(price.index, price.min(), price.max(),
                         where=ps > 0.5, color="red", alpha=0.15,
                         label="stress regime")
        ax1.legend(fontsize=8)
        ax2.plot(ps.index, ps.values, color="crimson", lw=1)
    ax2.axhline(0.5, color="k", ls="--", lw=0.7)
    ax2.set_ylabel("P(stress)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(alpha=0.3)
    return _save(fig, path)


# ----------------------------------------------------------------------
# (E) ML importances + fused score
# ----------------------------------------------------------------------
def plot_ml_importance(importance: pd.Series, path: str):
    s = importance.sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(s.index.astype(str), s.values, color="#2980b9")
    ax.set_title("(E) ML learning-to-rank — feature importance")
    ax.set_xlabel("importance")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def plot_screen_scores(screen: pd.DataFrame, path: str):
    comp = [c for c in ["value", "quality", "momentum", "value_x_mom",
                        "sentiment", "network", "ml"] if c in screen.columns]
    df = screen.sort_values("SCORE", ascending=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7),
                                   gridspec_kw={"width_ratios": [1, 2]})
    ax1.barh(df.index.astype(str), df["SCORE"].values,
             color=["#27ae60" if v >= 0 else "#c0392b" for v in df["SCORE"]])
    ax1.set_title("Fused SCORE (final ranking)")
    ax1.axvline(0, color="k", lw=0.7)
    df[comp].plot(kind="barh", stacked=True, ax=ax2, width=0.8, colormap="tab20")
    ax2.set_title("Score decomposition by model")
    ax2.axvline(0, color="k", lw=0.7)
    ax2.legend(fontsize=7, ncol=2)
    return _save(fig, path)


# ----------------------------------------------------------------------
# Markowitz allocation
# ----------------------------------------------------------------------
def plot_efficient_frontier(mu, Sig, tickers, w_ms, w_mv, rf, path):
    from markowitz import efficient_frontier, portfolio_return, portfolio_vol
    vols, rets, _ = efficient_frontier(mu, Sig, n_points=60)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, t in enumerate(tickers):
        ax.scatter(np.sqrt(Sig[i, i]), mu[i], s=25)
        ax.annotate(t, (np.sqrt(Sig[i, i]), mu[i]), fontsize=7)
    ax.plot(vols, rets, "b-", lw=2, label="Efficient frontier")
    ax.scatter(portfolio_vol(w_ms, Sig), portfolio_return(w_ms, mu),
               c="red", marker="*", s=280, label="Max Sharpe")
    ax.scatter(portfolio_vol(w_mv, Sig), portfolio_return(w_mv, mu),
               c="green", marker="*", s=280, label="Min Variance")
    ax.set_xlabel("Volatility (annualized)")
    ax.set_ylabel("Expected return (annualized)")
    ax.set_title("Markowitz efficient frontier")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_ic_summary(summary: pd.DataFrame, path: str):
    """Bar chart of each factor's annualized IC IR (signal consistency)."""
    s = summary["ic_ir"].sort_values()
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in s.values]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(s.index.astype(str), s.values, color=colors)
    ax.axvline(0, color="k", lw=0.7)
    ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    ax.axvline(-0.5, color="gray", ls="--", lw=0.8)
    ax.set_title("Factor IC IR (annualized) — bars past ±0.5 dashed line are useful")
    ax.set_xlabel("IC Information Ratio")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def plot_quantiles(quantiles: dict, path: str, n_q: int = 5):
    """Grid of quantile forward-return bars — monotonic slope = good factor."""
    names = list(quantiles)
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names):
        qr = quantiles[name]
        qs = [f"Q{q}" for q in range(1, n_q + 1)]
        vals = [qr.get(q, np.nan) * 100 for q in qs]
        ax.bar(qs, vals, color="#2980b9")
        ax.axhline(0, color="k", lw=0.6)
        ls = qr.get("long_short", np.nan) * 100
        ax.set_title(f"{name}  (L/S {ls:+.2f}%)", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle("Quantile forward returns (Q1=low factor .. Q5=high) — "
                 "rising bars = predictive", fontsize=11)
    return _save(fig, path)


def plot_backtest(strat_ret: pd.Series, bench_ret: pd.Series, path: str):
    """Equity curves (log) + drawdown for strategy vs benchmark."""
    def curve(r):
        return (1 + r.fillna(0)).cumprod()
    se, be = curve(strat_ret), curve(bench_ret)
    dd = se / se.cummax() - 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   height_ratios=[2, 1])
    ax1.plot(se.index, se.values, label="Strategy", color="navy", lw=1.5)
    ax1.plot(be.index, be.values, label="Equal-weight benchmark",
             color="gray", lw=1.2, ls="--")
    ax1.set_yscale("log")
    ax1.set_title("Walk-forward backtest — growth of 1 (log scale)")
    ax1.set_ylabel("equity (log)")
    ax1.legend(); ax1.grid(alpha=0.3, which="both")

    ax2.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    ax2.set_ylabel("drawdown")
    ax2.set_title("Strategy drawdown")
    ax2.grid(alpha=0.3)
    return _save(fig, path)


def plot_weights(weights, tickers, path, title="Portfolio weights"):
    pairs = [(t, w) for t, w in zip(tickers, weights) if w > 0.005]
    pairs.sort(key=lambda x: -x[1])
    labels, vals = zip(*pairs) if pairs else ([], [])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90,
           textprops={"fontsize": 9})
    ax.set_title(title)
    return _save(fig, path)


# ----------------------------------------------------------------------
# Orchestrator — live data, renders everything
# ----------------------------------------------------------------------
def make_all(outdir: str = FIGDIR):
    from data import download_prices, returns_stats
    from screener import run_screen, top_n
    from ml_rank import feature_importance
    from regime import regime_probabilities, _market_returns
    from markowitz import min_variance
    from scipy.optimize import minimize
    from markowitz import sharpe_ratio

    os.makedirs(outdir, exist_ok=True)
    paths = []

    screen, regime = run_screen()
    universe = list(screen.index)
    prices = download_prices(universe)

    paths.append(plot_price_history(prices, f"{outdir}/01_prices.png"))
    paths.append(plot_correlation_heatmap(prices, f"{outdir}/02_correlation.png"))
    paths.append(plot_factor_scores(screen, f"{outdir}/03_factors.png"))
    paths.append(plot_sentiment(screen["sentiment"], f"{outdir}/04_sentiment.png"))
    paths.append(plot_network(prices, f"{outdir}/05_network.png"))

    spy = _market_returns("SPY", "2015-01-01")
    if spy is not None:
        paths.append(plot_regime(spy, regime_probabilities(spy),
                                 f"{outdir}/06_regime.png"))
    paths.append(plot_ml_importance(feature_importance(prices),
                                    f"{outdir}/07_ml_importance.png"))
    paths.append(plot_screen_scores(screen, f"{outdir}/08_screen_scores.png"))

    # allocation on the top-N picks
    picks = top_n(screen, 10)
    pp = download_prices(picks)
    mu, Sig, tickers = returns_stats(pp)
    rf = 0.04
    n = len(tickers)
    w_ms = minimize(lambda w: -sharpe_ratio(w, mu, Sig, rf),
                    np.repeat(1/n, n), method="SLSQP",
                    bounds=[(0, 0.25)] * n,
                    constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}]).x
    w_mv = min_variance(Sig)
    paths.append(plot_efficient_frontier(mu, Sig, tickers, w_ms, w_mv, rf,
                                         f"{outdir}/09_frontier.png"))
    paths.append(plot_weights(w_ms, tickers, f"{outdir}/10_weights.png",
                              "Max-Sharpe portfolio weights"))

    print(f"Wrote {len(paths)} figures to ./{outdir}/")
    for p in paths:
        print("  ", p)
    return paths


def make_portfolio_charts(outdir="figures_sectorneutral",
                          prices_pkl="prices_universe.pkl",
                          top_n=15, max_per_sector=3, rf=0.04, cap=0.20):
    """
    Full chart set for the SECTOR-NEUTRAL universe portfolio, into `outdir`.
    Reuses cached universe prices (no re-download) for everything except the
    market-regime chart (SPY) and news sentiment (small live pull, best-effort).
    """
    import pandas as pd
    from scipy.optimize import minimize
    from data import returns_stats
    from factor_analysis import factor_panels
    from sector_select import load_sectors, sector_neutralize, select_sector_capped
    from ml_rank import feature_importance
    from markowitz import min_variance, sharpe_ratio
    from regime import detect_regime, regime_probabilities, _market_returns
    from backtester import walk_forward, benchmark_equal_weight, weight_max_sharpe
    from utils import zscore

    os.makedirs(outdir, exist_ok=True)
    prices_all = pd.read_pickle(prices_pkl).dropna(how="all")
    sectors = load_sectors()
    paths = []

    # --- build the sector-neutral portfolio ---
    panels = factor_panels(prices_all)
    mom = zscore(panels["momentum_12_1"].iloc[-1].dropna())
    rev = zscore(panels["reversal"].iloc[-1].dropna())
    raw = zscore(mom.add(rev, fill_value=0))
    neutral = sector_neutralize(raw, sectors)
    picks = select_sector_capped(neutral, sectors, top_n, max_per_sector)
    pp = prices_all[picks].dropna()

    def add(fn, name):
        try:
            paths.append(fn());
        except Exception as e:
            print(f"  (skipped {name}: {type(e).__name__}: {e})")

    # 01-02 prices & correlation of the picks
    add(lambda: plot_price_history(pp, f"{outdir}/01_prices.png"), "prices")
    add(lambda: plot_correlation_heatmap(pp, f"{outdir}/02_correlation.png"), "correlation")

    # 03 price-based factor scores for the picks
    ftab = pd.DataFrame({
        "momentum": mom.reindex(picks),
        "reversal": rev.reindex(picks),
        "low_vol":  zscore(panels["low_vol_63"].iloc[-1]).reindex(picks),
        "dist_high": zscore(panels["dist_high"].iloc[-1]).reindex(picks),
    }).fillna(0.0)
    add(lambda: plot_factor_scores(ftab, f"{outdir}/03_factor_scores.png",
                                   cols=["momentum", "reversal", "low_vol", "dist_high"]),
        "factor scores")

    # 04 sentiment (best-effort live news)
    def _sent():
        from sentiment import sentiment_scores
        return plot_sentiment(sentiment_scores(picks).reindex(picks).fillna(0.0),
                              f"{outdir}/04_sentiment.png")
    add(_sent, "sentiment")

    # 05 network of the picks
    add(lambda: plot_network(pp, f"{outdir}/05_network.png", threshold=0.3), "network")

    # 06 regime (market-level)
    def _regime():
        spy = _market_returns("SPY", "2015-01-01")
        return plot_regime(spy, regime_probabilities(spy), f"{outdir}/06_regime.png")
    add(_regime, "regime")

    # 07 ML feature importance (universe)
    add(lambda: plot_ml_importance(feature_importance(prices_all),
                                   f"{outdir}/07_ml_importance.png"), "ml importance")

    # 08 composite selection score (momentum + reversal, sector-neutralized)
    sdf = pd.DataFrame({"momentum": mom.reindex(picks),
                        "reversal": rev.reindex(picks)}).fillna(0.0)
    sdf["SCORE"] = neutral.reindex(picks)
    add(lambda: plot_screen_scores(sdf, f"{outdir}/08_scores.png"), "scores")

    # 09-10 Markowitz distribution
    mu, Sig, tickers = returns_stats(pp)
    n = len(tickers)
    w_ms = minimize(lambda w: -sharpe_ratio(w, mu, Sig, rf), np.repeat(1/n, n),
                    method="SLSQP", bounds=[(0, cap)] * n,
                    constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}]).x
    w_mv = min_variance(Sig)
    add(lambda: plot_efficient_frontier(mu, Sig, tickers, w_ms, w_mv, rf,
                                        f"{outdir}/09_frontier.png"), "frontier")
    add(lambda: plot_weights(w_ms, tickers, f"{outdir}/10_weights.png",
                             "Max-Sharpe (sector-neutral picks)"), "weights (max-sharpe)")
    add(lambda: plot_weights(w_mv, tickers, f"{outdir}/11_weights_minvar.png",
                             "Min-Variance (sector-neutral picks)"), "weights (min-var)")

    # 12 walk-forward backtest of the sector-neutral STRATEGY
    def _bt():
        def sn_score(window):
            wp = factor_panels(window)
            m = zscore(wp["momentum_12_1"].iloc[-1].dropna())
            r = zscore(wp["reversal"].iloc[-1].dropna())
            return sector_neutralize(zscore(m.add(r, fill_value=0)), sectors)
        sel = lambda s: select_sector_capped(s, sectors, top_n, max_per_sector)
        res = walk_forward(prices_all, sn_score, lambda w, p: weight_max_sharpe(w, p),
                           top_n=top_n, lookback=504, rebalance=21, cost_bps=10,
                           select_fn=sel)
        _, bench = benchmark_equal_weight(prices_all, start_from=504)
        bench = bench.reindex(res["returns"].index).fillna(0.0)
        return plot_backtest(res["returns"], bench, f"{outdir}/12_backtest.png")
    add(_bt, "backtest")

    print(f"Wrote {len(paths)} figures to ./{outdir}/  (portfolio: {picks})")
    return paths


if __name__ == "__main__":
    import sys
    if "--portfolio" in sys.argv:
        make_portfolio_charts()
    else:
        make_all()
