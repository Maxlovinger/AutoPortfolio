"""
Tests for point-in-time membership reconstruction. The reconstruction logic is
pure, so these run fully offline with synthetic changes tables. Network scraping
(parse_current/parse_changes) is exercised against a saved-shape fixture.
"""
import numpy as np
import pandas as pd
import pytest

from historical_membership import (
    members_on, membership_snapshots, universe_on, delisted_members,
    price_coverage, parse_changes, parse_current, _clean_opt,
    restrict_to_members, pit_score,
)


@pytest.fixture
def changes():
    # A was added 2020, B removed 2021, C added 2022
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-06-01", "2021-06-01", "2022-06-01"]),
        "added": ["A", "", "C"],
        "removed": ["", "B", ""],
    })


def test_members_today_is_current(changes):
    current = {"A", "C", "D"}
    assert members_on("2026-01-01", current, changes) == {"A", "C", "D"}


def test_rewind_before_last_add(changes):
    # before C was added (2022), C should not be a member
    current = {"A", "C", "D"}
    m = members_on("2022-01-01", current, changes)
    assert "C" not in m
    assert m == {"A", "D"}


def test_rewind_restores_removed(changes):
    # before B was removed (2021) but after A added (2020): B back, C gone
    current = {"A", "C", "D"}
    m = members_on("2021-01-01", current, changes)
    assert "B" in m and "C" not in m
    assert m == {"A", "B", "D"}


def test_rewind_before_everything(changes):
    # before A added (2020): no A, no C, B present
    current = {"A", "C", "D"}
    m = members_on("2019-01-01", current, changes)
    assert m == {"B", "D"}


def test_snapshots_shape_and_pit(changes):
    current = {"A", "C", "D"}
    snap = membership_snapshots(current, changes, start="2019-01-01",
                                end="2023-01-01", freq="YS")
    assert set(snap.columns) == {"date", "ticker"}
    # earliest snapshot must contain B (delisted name) — the whole point
    first = snap["date"].min()
    assert "B" in set(snap.loc[snap["date"] == first, "ticker"])


def test_universe_on_uses_prior_snapshot(changes):
    current = {"A", "C", "D"}
    snap = membership_snapshots(current, changes, start="2019-01-01",
                                end="2023-01-01", freq="YS")
    u = universe_on("2019-06-15", snap)   # between 2019 and 2020 snapshots
    assert "B" in u and "C" not in u


def test_delisted_members_flags_dead_names(changes):
    snap = membership_snapshots({"A", "C", "D"}, changes,
                                start="2019-01-01", end="2023-01-01", freq="YS")
    dead = delisted_members(snap, current_universe={"A", "C", "D"})
    assert dead == {"B"}


def test_price_coverage_reports_gap():
    snap = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"] * 3),
                         "ticker": ["A", "B", "C"]})
    prices = pd.DataFrame({"A": [1.0], "C": [1.0]})   # B missing
    cov = price_coverage(snap, prices)
    assert cov["members"] == 3
    assert cov["have_prices"] == 2
    assert cov["missing_prices"] == 1
    assert cov["coverage"] == pytest.approx(2 / 3)


def test_restrict_to_members_enforces_pit(changes):
    snap = membership_snapshots({"A", "C", "D"}, changes,
                                start="2019-01-01", end="2023-01-01", freq="YS")
    window = pd.DataFrame({"A": [1, 2], "B": [1, 2], "C": [1, 2], "D": [1, 2]},
                          index=pd.to_datetime(["2019-05-01", "2019-06-01"]))
    r = restrict_to_members(window, snap)
    # in mid-2019 B is a member, C is not yet
    assert "B" in r.columns and "C" not in r.columns


def test_pit_score_drops_nonmembers(changes):
    snap = membership_snapshots({"A", "C", "D"}, changes,
                                start="2019-01-01", end="2023-01-01", freq="YS")
    window = pd.DataFrame({"A": [1, 2], "B": [1, 2], "C": [1, 2]},
                          index=pd.to_datetime(["2019-05-01", "2019-06-01"]))
    scored = pit_score(lambda w: pd.Series(1.0, index=w.columns), snap)(window)
    assert "C" not in scored.index and "B" in scored.index


def test_clean_opt_handles_nan_and_dots():
    assert _clean_opt(np.nan) == ""
    assert _clean_opt("nan") == ""
    assert _clean_opt("BRK.B") == "BRK-B"
    assert _clean_opt(" aapl ") == "AAPL"


def test_parse_current_and_changes_on_synthetic_tables():
    current_tbl = pd.DataFrame({"Symbol": ["AAA", "BBB"],
                                "Security": ["A Inc", "B Inc"]})
    changes_tbl = pd.DataFrame({
        ("Date", "Date"): ["June 1, 2021"],
        ("Added", "Ticker"): ["AAA"],
        ("Added", "Security"): ["A Inc"],
        ("Removed", "Ticker"): ["ZZZ"],
        ("Removed", "Security"): ["Z Inc"],
    })
    changes_tbl.columns = pd.MultiIndex.from_tuples(changes_tbl.columns)
    tables = [current_tbl, changes_tbl]
    assert parse_current(tables) == {"AAA", "BBB"}
    ch = parse_changes(tables)
    assert list(ch["added"]) == ["AAA"]
    assert list(ch["removed"]) == ["ZZZ"]
    assert ch["date"].iloc[0] == pd.Timestamp("2021-06-01")
