"""
Tests for fx_execution.py — the FX carry sleeve's order translation. The whole
point of this module is getting IBKR's per-pair quote convention right, so these
pin down: pair orientation, side/action, quantity units, and the two friction
flags (odd-lot below IDEALPRO min, NDF-skip). A fake app/ibkr verifies routing
respects dry-run and never sends NDF legs.
"""
import numpy as np
import pandas as pd
import pytest

import fx_execution as fxe


# --- pair orientation ------------------------------------------------------
def test_ib_pair_orientation():
    # invert=False majors trade as FOREIGN.USD
    assert fxe.ib_pair("EUR") == ("EUR", "USD")
    assert fxe.ib_pair("AUD") == ("AUD", "USD")
    # invert=True (JPY, CHF, EM...) trade as USD.FOREIGN
    assert fxe.ib_pair("MXN") == ("USD", "MXN")
    assert fxe.ib_pair("CHF") == ("USD", "CHF")
    with pytest.raises(ValueError):
        fxe.ib_pair("USD")


# --- the convention trap: long/short map to the right side -----------------
def test_long_em_foreign_is_sell_usd_pair():
    # LONG MXN $10k: pair USD.MXN, quantity in USD, side SELL (sell USD/buy MXN)
    leg = fxe.plan_leg("MXN", +10_000, spot_px=0.058)
    assert leg["pair"] == "USD.MXN"
    assert leg["action"] == "SELL" and leg["side"] == "LONG"
    assert leg["qty"] == 10_000              # USD (base) units


def test_short_em_foreign_is_buy_usd_pair():
    # SHORT CHF $10k: pair USD.CHF, side SHORT, action BUY the pair
    leg = fxe.plan_leg("CHF", -10_000, spot_px=1.25)
    assert leg["pair"] == "USD.CHF"
    assert leg["action"] == "BUY" and leg["side"] == "SHORT"
    assert leg["qty"] == 10_000


def test_long_major_foreign_is_buy_and_qty_in_foreign():
    # LONG EUR $10k at 1.08 USD/EUR: pair EUR.USD, BUY, qty = 10000/1.08 EUR
    leg = fxe.plan_leg("EUR", +10_000, spot_px=1.08)
    assert leg["pair"] == "EUR.USD"
    assert leg["action"] == "BUY" and leg["side"] == "LONG"
    assert leg["qty"] == round(10_000 / 1.08)


def test_short_major_is_sell():
    leg = fxe.plan_leg("AUD", -5_000, spot_px=0.66)
    assert leg["action"] == "SELL" and leg["side"] == "SHORT"
    assert leg["qty"] == round(5_000 / 0.66)


# --- guards ----------------------------------------------------------------
def test_zero_and_bad_price_return_none():
    assert fxe.plan_leg("MXN", 0.0, 0.058) is None
    assert fxe.plan_leg("MXN", 10_000, np.nan) is None
    assert fxe.plan_leg("MXN", 10_000, 0.0) is None


def test_below_min_flag():
    small = fxe.plan_leg("MXN", 1_740, 0.058)         # small book per-leg size
    big = fxe.plan_leg("MXN", 40_000, 0.058)
    assert small["below_min"] is True
    assert big["below_min"] is False


def test_ndf_currencies_flagged_not_tradeable():
    leg = fxe.plan_leg("KRW", 10_000, 0.00075)
    assert leg["tradeable"] is False       # KRW is NDF -> not spot-tradeable


# --- full plan -------------------------------------------------------------
def _spot():
    return pd.Series({"MXN": 0.058, "HUF": 0.0028, "ZAR": 0.055,
                      "CHF": 1.25, "CAD": 0.73, "SEK": 0.095, "EUR": 1.08})


def test_fx_order_plan_preserves_gross_and_neutrality():
    targets = {"MXN": 5_000, "HUF": 5_000, "ZAR": 5_000,
               "CHF": -5_000, "CAD": -5_000, "SEK": -5_000}
    plan = fxe.fx_order_plan(targets, _spot())
    s = fxe.summarize_plan(plan)
    assert s["n_legs"] == 6
    assert s["net_usd"] == pytest.approx(0.0, abs=1.0)     # dollar-neutral
    assert s["gross_usd"] == pytest.approx(30_000, abs=1.0)
    # longs come first after sort
    assert plan[0]["side"] == "LONG"


def test_fx_order_plan_skips_missing_spot():
    targets = {"MXN": 5_000, "XXX": 5_000}
    plan = fxe.fx_order_plan(targets, _spot())            # XXX not in spot
    assert [l["ccy"] for l in plan] == ["MXN"]


# --- routing (fake app / ibkr) ---------------------------------------------
class _FakeApp:
    def __init__(self):
        self._oid = 1
        self.order_status = {}
        self.placed = []

    def next_order_id(self):
        oid = self._oid
        self._oid += 1
        return oid

    def placeOrder(self, oid, contract, order):
        self.placed.append((oid, contract, order))
        self.order_status[oid] = {"status": "Submitted"}


class _FakeIbkr:
    def forex(self, base, quote="USD"):
        return (base, quote)

    def market_order(self, action, qty):
        return (action, qty)


def test_route_fx_dry_run_sends_nothing():
    plan = fxe.fx_order_plan({"MXN": 5_000, "CHF": -5_000}, _spot())
    app = _FakeApp()
    out = fxe.route_fx(app, plan, _FakeIbkr(), dry_run=True)
    assert app.placed == []
    assert all(l["status"] == "DRY-RUN" for l in out)


def test_route_fx_live_sends_tradeable_only():
    targets = {"MXN": 5_000, "KRW": 5_000, "CHF": -5_000}   # KRW is NDF
    spot = _spot()
    spot["KRW"] = 0.00075
    plan = fxe.fx_order_plan(targets, spot)
    app = _FakeApp()
    fxe.route_fx(app, plan, _FakeIbkr(), dry_run=False)
    sent = {c for (_, (base, quote), _) in app.placed for c in (base, quote)}
    assert "MXN" in sent and "CHF" in sent
    assert "KRW" not in sent                                # NDF never transmitted
    krw = next(l for l in plan if l["ccy"] == "KRW")
    assert krw["status"] == "SKIPPED-NDF"
