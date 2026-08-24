"""Tests for the FX carry sleeve's reconciliation core (run_fx.held_to_signed /
reconcile_delta) — the logic that must invert IB's per-pair quote convention
correctly so a monthly re-run nets against holdings instead of stacking them."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_fx


SPOT = pd.Series({"EUR": 1.08, "JPY": 1 / 159.0, "CHF": 1.25, "HUF": 1 / 310.0})


def test_usd_foreign_pair_long_usd_is_short_foreign():
    # held USD.JPY +2175 (long USD) => SHORT JPY 2175 USD
    cur = run_fx.held_to_signed([("USD", "JPY", 2175.0)], SPOT)
    assert cur == {"JPY": -2175.0}


def test_usd_foreign_pair_short_usd_is_long_foreign():
    # held USD.HUF -2175 (short USD) => LONG HUF 2175 USD
    cur = run_fx.held_to_signed([("USD", "HUF", -2175.0)], SPOT)
    assert cur == {"HUF": 2175.0}


def test_foreign_usd_pair_uses_spot():
    # held EUR.USD +2000 EUR at 1.08 => LONG EUR 2160 USD
    cur = run_fx.held_to_signed([("EUR", "USD", 2000.0)], SPOT)
    assert cur["EUR"] == pytest.approx(2160.0)


def test_foreign_usd_pair_missing_spot_skipped():
    cur = run_fx.held_to_signed([("SEK", "USD", 1000.0)], SPOT)  # SEK not in SPOT
    assert cur == {}


def test_reconcile_delta_nets_against_current():
    target = {"CHF": -2175, "HUF": 2175, "JPY": -2175, "MXN": 2175}
    current = {"CHF": -2175, "HUF": 2175}          # 2 legs already on target
    delta = run_fx.reconcile_delta(target, current)
    assert set(delta) == {"JPY", "MXN"}            # only the missing legs remain
    assert delta["JPY"] == -2175 and delta["MXN"] == 2175


def test_reconcile_delta_idempotent_when_on_target():
    tgt = {"CHF": -2175, "HUF": 2175}
    assert run_fx.reconcile_delta(tgt, dict(tgt)) == {}   # nothing to trade


def test_reconcile_delta_no_churn_band():
    # a $20 drift is below the $50 band -> ignored
    assert run_fx.reconcile_delta({"CHF": -2195}, {"CHF": -2175}, min_usd=50.0) == {}
    # a $200 drift trades
    assert run_fx.reconcile_delta({"CHF": -2375}, {"CHF": -2175}, min_usd=50.0) == {"CHF": -200.0}


def test_roundtrip_establish_then_reconcile_is_flat():
    """Establish a book, 'hold' exactly it, re-reconcile -> zero (cron-safe)."""
    target = {"JPY": -2175.0, "HUF": 2175.0}
    # simulate the fills as IB would report them (USD.JPY long, USD.HUF short)
    raw = [("USD", "JPY", 2175.0), ("USD", "HUF", -2175.0)]
    current = run_fx.held_to_signed(raw, SPOT)
    assert run_fx.reconcile_delta(target, current) == {}
