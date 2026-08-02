"""Tests for sector-neutral ranking + sector-capped selection."""
import numpy as np
import pandas as pd
import pytest
from sector_select import (
    sector_neutralize, select_sector_capped, sector_breakdown,
)

# 6 names across 2 sectors; Tech scores all high, Fin all low
SCORES = pd.Series({"T1": 5.0, "T2": 4.0, "T3": 3.0, "F1": 1.0, "F2": 0.5, "F3": 0.0})
SECTORS = {"T1": "Tech", "T2": "Tech", "T3": "Tech",
           "F1": "Fin", "F2": "Fin", "F3": "Fin"}


def test_neutralize_zero_mean_within_sector():
    z = sector_neutralize(SCORES, SECTORS)
    tech = z[["T1", "T2", "T3"]]
    fin = z[["F1", "F2", "F3"]]
    assert abs(tech.mean()) < 1e-9 and abs(fin.mean()) < 1e-9


def test_neutralize_makes_best_in_sector_comparable():
    # raw: all Tech > all Fin. After neutralizing, best Fin (F1) should rank
    # as high as best Tech (T1) since each is top of its own sector.
    z = sector_neutralize(SCORES, SECTORS)
    assert z["F1"] == pytest.approx(z["T1"], abs=1e-9)


def test_sector_cap_limits_per_sector():
    picks = select_sector_capped(SCORES, SECTORS, top_n=4, max_per_sector=2)
    counts = sector_breakdown(picks, SECTORS)
    assert all(c <= 2 for c in counts.values())
    assert len(picks) == 4


def test_sector_cap_without_neutralize_still_diversifies():
    # even on raw (Tech-dominant) scores, cap forces a Fin name in
    picks = select_sector_capped(SCORES, SECTORS, top_n=4, max_per_sector=2)
    assert any(SECTORS[p] == "Fin" for p in picks)


def test_neutralized_selection_balances_sectors():
    z = sector_neutralize(SCORES, SECTORS)
    picks = select_sector_capped(z, SECTORS, top_n=2, max_per_sector=1)
    assert set(sector_breakdown(picks, SECTORS)) == {"Tech", "Fin"}


def test_unknown_sector_handled():
    scores = pd.Series({"A": 1.0, "B": 2.0})
    picks = select_sector_capped(scores, {}, top_n=2, max_per_sector=5)  # no labels
    assert picks == ["B", "A"]


def test_fewer_names_than_top_n():
    picks = select_sector_capped(SCORES, SECTORS, top_n=99, max_per_sector=99)
    assert len(picks) == 6
