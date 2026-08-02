"""
screener.py — the orchestrator that FUSES all five models into one ranking.

    (A) factors        -> value / quality / momentum / interaction composite
    (B) sentiment      -> news-tone overlay
    (C) network        -> diversifier (low-centrality) tilt
    (D) regime         -> sets the WEIGHTS on everything above (adaptive)
    (E) ml_rank        -> gradient-boosted forward-return prediction (blended small)

Output: a ranked DataFrame of stocks. The TOP-N becomes the universe handed to
the Markowitz optimizer (run.py).

This is the "alpha" stage. Markowitz then decides the weights among the winners.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from data import download_prices
from factors import build_factor_table
from sentiment import sentiment_scores
from network_model import network_scores, community_labels
from regime import detect_regime, regime_factor_weights
from ml_rank import ml_rank_scores
from utils import zscore


# Diversified candidate universe across sectors (a LEARNING universe).
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",      # tech
    "JPM", "BAC", "V",                            # financials
    "JNJ", "UNH", "LLY",                          # healthcare
    "XOM", "CVX",                                 # energy
    "PG", "KO", "COST",                           # staples
    "HD", "MCD", "NKE",                           # discretionary
    "CAT", "HON",                                 # industrials
    "NEE", "DUK",                                 # utilities
    "AMT",                                        # real estate
]

ML_BLEND = 0.25  # small weight on the ML model (overfitting risk)


def run_screen(universe=None, start="2018-01-01", verbose=True):
    universe = universe or DEFAULT_UNIVERSE
    prices = download_prices(universe, start=start)
    universe = list(prices.columns)  # keep only tickers that downloaded
    if verbose:
        print(f"Screening {len(universe)} stocks over "
              f"{prices.index[0].date()} -> {prices.index[-1].date()}")

    # (D) regime first — it decides the weights on everything else
    regime = detect_regime()
    w = regime_factor_weights(regime)
    if verbose:
        print(f"Regime: {regime['regime'].upper()} "
              f"(p_stress={regime['p_stress']:.2f}, method={regime['method']})")

    # (A) factor composite
    ftable = build_factor_table(prices, universe)
    # (B) sentiment, (C) network
    ftable["sentiment"] = sentiment_scores(universe).reindex(universe).fillna(0.0)
    ftable["network"]   = network_scores(prices).reindex(universe).fillna(0.0)

    # weighted alpha score using regime-adaptive weights
    alpha = sum(ftable[k] * w.get(k, 0.0) for k in
                ["value", "quality", "momentum", "value_x_mom",
                 "sentiment", "network"])
    alpha = zscore(alpha)

    # (E) ML ranker, blended in
    ml = ml_rank_scores(prices).reindex(universe).fillna(0.0)
    final = zscore((1 - ML_BLEND) * alpha + ML_BLEND * ml)

    out = ftable.copy()
    out["ml"] = ml
    out["alpha"] = alpha
    out["SCORE"] = final
    out["community"] = pd.Series(community_labels(prices)).reindex(universe)
    out = out.sort_values("SCORE", ascending=False)
    return out, regime


def top_n(screen_df: pd.DataFrame, n: int = 10, one_per_community=False):
    """Pick the top-N. Optionally cap to diversify across correlation clusters."""
    if not one_per_community:
        return list(screen_df.head(n).index)
    picked, seen = [], set()
    for t, row in screen_df.iterrows():
        c = row.get("community")
        if c in seen:
            continue
        picked.append(t); seen.add(c)
        if len(picked) >= n:
            break
    return picked


if __name__ == "__main__":
    df, regime = run_screen()
    cols = ["value", "quality", "momentum", "value_x_mom",
            "sentiment", "network", "ml", "SCORE"]
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== FACTOR SCREEN (ranked) ===")
    print(df[cols].round(2))
    print("\nTop 10 picks ->", top_n(df, 10))
