"""
sector_select.py — sector-neutral ranking + sector-capped selection.

Problem it fixes: plain factor ranking piles into whatever sector is hottest
(our momentum picks were ~all semiconductors). Two standard remedies:

  1. SECTOR-NEUTRALIZE: z-score each factor WITHIN its sector, so a stock is
     judged against its peers, not against a different industry. This removes
     the "whole sector looks good" bias.
  2. SECTOR CAP: when selecting the top names, allow at most `max_per_sector`
     from any one sector, forcing spread across industries.

Sector labels come from universe.csv (built by universe.py).
"""
from __future__ import annotations
import pandas as pd
from utils import zscore


def load_sectors(path="universe.csv") -> dict:
    df = pd.read_csv(path, index_col=0)
    col = "sector" if "sector" in df.columns else df.columns[0]
    return df[col].fillna("Unknown").astype(str).to_dict()


def sector_neutralize(scores: pd.Series, sectors: dict) -> pd.Series:
    """Z-score the factor WITHIN each sector (comparable across sectors)."""
    sec = pd.Series({t: sectors.get(t, "Unknown") for t in scores.index})
    return scores.groupby(sec).transform(zscore).reindex(scores.index)


def select_sector_capped(scores: pd.Series, sectors: dict,
                         top_n: int = 15, max_per_sector: int = 3) -> list:
    """Highest scores first, but at most `max_per_sector` from any one sector."""
    picks, counts = [], {}
    for t in scores.sort_values(ascending=False).index:
        sec = sectors.get(t, "Unknown")
        if counts.get(sec, 0) >= max_per_sector:
            continue
        picks.append(t)
        counts[sec] = counts.get(sec, 0) + 1
        if len(picks) >= top_n:
            break
    return picks


def sector_breakdown(picks: list, sectors: dict) -> dict:
    out = {}
    for t in picks:
        s = sectors.get(t, "Unknown")
        out[s] = out.get(s, 0) + 1
    return dict(sorted(out.items(), key=lambda x: -x[1]))
