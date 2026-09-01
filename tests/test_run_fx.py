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


# --- ledger_to_signed: the reliable source (reqPositions can't see spot-FX) --------

def test_ledger_long_foreign_positive_usd():
    # +1,042,158 ZAR at 0.0620553 USD/ZAR => LONG ZAR ~ +$64,671
    cur = run_fx.ledger_to_signed({"ZAR": {"cash": 1042157.82, "rate": 0.0620553}})
    assert cur["ZAR"] == pytest.approx(64671.0, abs=5)


def test_ledger_short_foreign_negative_usd():
    # -52,289.92 CHF at 1.2352072 USD/CHF => SHORT CHF ~ -$64,589
    cur = run_fx.ledger_to_signed({"CHF": {"cash": -52289.92, "rate": 1.2352072}})
    assert cur["CHF"] == pytest.approx(-64589.0, abs=5)


def test_ledger_excludes_usd_and_base():
    led = {"USD": {"cash": 64604.0, "rate": 1.0}, "BASE": {"cash": 64687.0, "rate": 1.0},
           "ZAR": {"cash": 1000000.0, "rate": 0.062}}
    assert set(run_fx.ledger_to_signed(led)) == {"ZAR"}


def test_ledger_skips_dust_below_band():
    # accrual dust (0 cash) and a tiny balance are dropped
    led = {"JPY": {"cash": 0.0, "rate": 0.0062}, "MXN": {"cash": 100.0, "rate": 0.0588}}
    assert run_fx.ledger_to_signed(led, min_usd=50.0) == {}   # $0 and ~$5.9 both < $50


def test_ledger_skips_incomplete_rows():
    led = {"CHF": {"cash": -52289.92}, "ZAR": {"rate": 0.062}}  # missing rate / cash
    assert run_fx.ledger_to_signed(led) == {}


def test_ledger_reconcile_nets_and_does_not_stack():
    """Regression for the doubling bug: with the sleeve already held (visible only in
    the ledger), reconciling against target must UNWIND the excess, not re-add it."""
    target = {"CHF": -32331.0, "ZAR": 32331.0}                  # 1x sleeve
    held = {"CHF": {"cash": -52289.92, "rate": 1.2352072},      # ~2x currently held
            "ZAR": {"cash": 1042157.82, "rate": 0.0620553}}
    current = run_fx.ledger_to_signed(held)
    delta = run_fx.reconcile_delta(target, current, min_usd=50.0)
    # trim back toward 1x: buy CHF (cover part of the short), sell ZAR (cut the long)
    assert delta["CHF"] > 0 and delta["ZAR"] < 0
    assert current["CHF"] + delta["CHF"] == pytest.approx(target["CHF"])
    assert current["ZAR"] + delta["ZAR"] == pytest.approx(target["ZAR"])


def test_ledger_flat_book_builds_full_target():
    """A genuinely flat book (ledger has only USD/BASE) yields current={} so the full
    target is traded once — the correct behavior the abort-guard protects."""
    assert run_fx.ledger_to_signed({"USD": {"cash": 100000.0, "rate": 1.0}}) == {}
