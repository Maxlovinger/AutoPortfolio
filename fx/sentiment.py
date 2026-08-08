"""
fx/sentiment.py — news-sentiment signal for the currency sleeve.

The FX analogue of the equity sentiment overlay: for each currency we take the
monthly TONE of that economy's news (GDELT country_tone — pre-computed,
timestamped, so backtestable), cross-sectionally z-score it, and expose it as a
signal the composite ranker can blend — OR test on its own vs plain carry.

No look-ahead: tone is month-end aggregated then SHIFTED one month (a month's
news informs next month's position), exactly like the carry alignment.

Data caveat: GDELT rate-limits/blocks datacenter IPs (HTTP 429 from the
sandbox); run the fetch from a normal IP. Results cache to .gdelt_cache so a
backtest hits the API once. This module is import-safe offline; only the
`*_from_gdelt` fetchers touch the network.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from fx.backtest_carry import FREQ, resample_spot, build_asset_returns, run_rank_backtest
from fx.composite import zscore_rows


def align_tone(tone_panel: pd.DataFrame, grid_index, shift=1) -> pd.DataFrame:
    """Forward-fill a monthly tone panel onto the trade grid and shift `shift`
    periods so weights use only prior-month news (no look-ahead)."""
    idx = grid_index
    aligned = (tone_panel.reindex(tone_panel.index.union(idx)).ffill()
                         .reindex(idx).shift(shift))
    return aligned


def sentiment_signal(spot, tone_panel, freq="M") -> pd.DataFrame:
    """Cross-sectionally z-scored, grid-aligned, no-look-ahead sentiment score
    per currency (higher = more positive economy news than peers)."""
    grid = resample_spot(spot, freq).pct_change().index
    aligned = align_tone(tone_panel, grid)
    return zscore_rows(aligned)


def run_sentiment_backtest(spot, carry, tone_panel, freq="M",
                           n_long=3, n_short=3, gross=1.0, cost_bps=5.0):
    """Rank purely on news sentiment (carry-inclusive returns still realized),
    so it's directly comparable to run_carry_backtest / composite."""
    asset_ret, _ = build_asset_returns(spot, carry, freq)
    score = sentiment_signal(spot, tone_panel, freq).reindex(
        asset_ret.index)[asset_ret.columns]
    return run_rank_backtest(asset_ret, score, freq, n_long, n_short,
                             gross, cost_bps)


def tone_panel_from_gdelt(ccys, timespan="48m") -> pd.DataFrame:
    """Fetch the monthly country-tone panel for the FX universe from GDELT.
    Network + cache; run from a non-datacenter IP (see module docstring)."""
    from news_gdelt import country_tone_panel
    return country_tone_panel(ccys, timespan=timespan)
