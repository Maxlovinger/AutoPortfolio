"""
Offline tests for the 3-sleeve live allocator (portfolio_live.py). The live()
data pull is integration; here the pure allocation math is pinned down:
sleeve weights sum right, the de-risked equity slice goes to cash (not bonds),
bonds are a fixed weight, and the FX book is dollar-neutral at the right gross.
"""
import numpy as np
import pandas as pd
import pytest

import portfolio_live as pl


def test_design_weights_sum_to_one():
    assert sum(pl.WEIGHTS.values()) == pytest.approx(1.0)
    # the locked split
    assert pl.WEIGHTS["equity"] == pytest.approx(0.639)
    assert pl.WEIGHTS["currency"] == pytest.approx(0.261)
    assert pl.WEIGHTS["bonds"] == pytest.approx(0.10)


def test_allocation_dollars():
    a = pl.allocation(100_000)
    assert a["equity"] == pytest.approx(63_900)
    assert a["currency"] == pytest.approx(26_100)
    assert a["bonds"] == pytest.approx(10_000)


def test_stock_sleeve_full_exposure():
    picks = [f"S{i}" for i in range(30)]
    w = pl.stock_sleeve_weights(picks, exposure=1.0)
    # 30 equal equity names summing to 63.9%, + IEF 10%
    assert w["IEF"] == pytest.approx(0.10)
    eq = w.drop("IEF")
    assert eq.sum() == pytest.approx(0.639)
    assert (eq == eq.iloc[0]).all()                     # equal weight
    assert w.sum() == pytest.approx(0.739)              # rest (currency+cash) elsewhere


def test_derisked_equity_goes_to_cash_not_bonds():
    picks = [f"S{i}" for i in range(30)]
    w = pl.stock_sleeve_weights(picks, exposure=0.6)
    # equity scaled by exposure -> 0.639*0.6; IEF UNCHANGED at 0.10
    assert w.drop("IEF").sum() == pytest.approx(0.639 * 0.6)
    assert w["IEF"] == pytest.approx(0.10)              # bonds NOT scaled by exposure
    # implied cash grows as equity de-risks (stock+currency < 1)
    implied_cash = 1 - w.sum() - pl.WEIGHTS["currency"]
    assert implied_cash > 0


def test_bond_only_when_no_equity_picks():
    w = pl.stock_sleeve_weights([], exposure=1.0)
    assert list(w.index) == ["IEF"] and w["IEF"] == pytest.approx(0.10)


def test_fx_sleeve_dollar_neutral_and_gross():
    carry = pd.Series({"AAA": 5.0, "BBB": 4.0, "CCC": 3.0, "DDD": 1.0,
                       "EEE": 0.5, "FFF": -2.0})
    fx = pl.fx_sleeve_targets(carry, nav=100_000, n_long=2, n_short=2)
    net = sum(fx.values())
    gross = sum(abs(v) for v in fx.values())
    assert net == pytest.approx(0.0, abs=1.0)           # dollar-neutral
    assert gross == pytest.approx(0.261 * 100_000, rel=1e-6)   # gross = currency sleeve
    # highest carry long, lowest short
    assert fx["AAA"] > 0 and fx["FFF"] < 0


def test_fx_targets_scale_with_nav():
    carry = pd.Series({"AAA": 5.0, "BBB": 4.0, "CCC": 0.0, "DDD": -3.0, "EEE": -4.0})
    small = pl.fx_sleeve_targets(carry, 10_000, n_long=2, n_short=2)
    big = pl.fx_sleeve_targets(carry, 100_000, n_long=2, n_short=2)
    for c in small:
        assert big[c] == pytest.approx(small[c] * 10, rel=1e-6)


def test_preview_returns_consistent_book(capsys):
    picks = [f"S{i}" for i in range(30)]
    carry = pd.Series({"AAA": 5.0, "BBB": 4.0, "CCC": 3.0, "DDD": 1.0,
                       "EEE": 0.5, "FFF": -2.0})
    out = pl.preview(40_000, picks, exposure=1.0, carry_row=carry)
    assert set(out) == {"stock_weights", "fx_targets", "allocation", "cash"}
    # at full exposure the whole book is invested -> ~0 cash
    assert abs(out["cash"]) < 1.0
    assert "3-SLEEVE TARGET BOOK" in capsys.readouterr().out
