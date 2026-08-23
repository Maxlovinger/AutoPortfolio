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


# ---------------------------------------------------------------------------
# greedy_share_topup — the whole-share rounding-drag redeployment. These pin
# down the exact behaviour that fixed the live book (equity 43.6% -> 56.7%).
# ---------------------------------------------------------------------------
def _floor_shares(targets, prices):
    """What plan_orders' whole-share floor would hold (the drag-leaving baseline)."""
    return {t: int(targets[t] // prices[t]) for t in targets}


def test_topup_never_sells_only_adds():
    targets = {"A": 1000.0, "B": 1000.0}
    prices = pd.Series({"A": 100.0, "B": 250.0})
    cur = {"A": 20, "B": 2}                      # A already OVER target ($2000)
    add = pl.greedy_share_topup(targets, cur, prices)
    assert all(v > 0 for v in add.values())      # only BUYs are returned
    assert "A" not in add                        # over-target name is left alone


def test_topup_fills_toward_equal_dollar_target():
    # four $1000 targets, all under-held; top-up should push each near $1000
    targets = {t: 1000.0 for t in "ABCD"}
    prices = pd.Series({"A": 100.0, "B": 200.0, "C": 250.0, "D": 500.0})
    cur = _floor_shares(targets, prices)          # A9=900,B4=800... wait: floors
    add = pl.greedy_share_topup(targets, cur, prices)
    final = {t: (cur.get(t, 0) + add.get(t, 0)) * prices[t] for t in targets}
    # every name ends within one share-price of its $1000 target (no big miss)
    for t in targets:
        assert final[t] >= 1000.0 - prices[t] - 1e-6
        assert final[t] <= 1000.0 * 1.5 + 1e-6    # per-name cap respected


def test_topup_recovers_a_dropped_zero_share_name():
    # C is priced ABOVE its target -> plan_orders floor holds 0 shares (dropped);
    # the top-up must still let it take its first share (the LLY/GS/GEV case).
    targets = {"A": 950.0, "B": 950.0, "C": 950.0}
    prices = pd.Series({"A": 90.0, "B": 90.0, "C": 1200.0})
    cur = {"A": 10, "B": 10, "C": 0}              # C dropped by the floor
    add = pl.greedy_share_topup(targets, cur, prices)
    assert add.get("C", 0) == 1                   # dropped name enters the book


def test_topup_respects_per_name_cap_for_held_names():
    # a cheap name already near target must not be inflated past the cap
    targets = {"A": 1000.0}
    prices = pd.Series({"A": 10.0})
    cur = {"A": 95}                               # $950, gap $50
    add = pl.greedy_share_topup(targets, cur, prices, per_name_cap=1.5)
    final = (95 + add.get("A", 0)) * 10.0
    assert final <= 1000.0 * 1.5 + 1e-6
    # it can buy up to the cap but not beyond
    assert add.get("A", 0) >= 1


def test_topup_budget_cap_is_not_exceeded():
    targets = {t: 1000.0 for t in "ABCDE"}
    prices = pd.Series({t: 100.0 for t in "ABCDE"})
    cur = {t: 0 for t in "ABCDE"}
    add = pl.greedy_share_topup(targets, cur, prices, budget=250.0)
    spent = sum(add[t] * prices[t] for t in add)
    assert spent <= 250.0 + 1e-6                  # never overspends the budget
    assert spent == pytest.approx(200.0)          # 2 whole $100 shares fit in $250


def test_topup_default_budget_is_total_gap():
    targets = {"A": 1000.0, "B": 1000.0}
    prices = pd.Series({"A": 100.0, "B": 100.0})
    cur = {"A": 5, "B": 5}                         # each $500, total gap $1000
    add = pl.greedy_share_topup(targets, cur, prices)   # budget=None -> gap
    spent = sum(add[t] * prices[t] for t in add)
    assert spent == pytest.approx(1000.0)         # exactly fills to target here


def test_topup_skips_bad_prices():
    targets = {"A": 1000.0, "BAD": 1000.0, "ZERO": 1000.0}
    prices = pd.Series({"A": 100.0, "BAD": np.nan, "ZERO": 0.0})
    cur = {"A": 0, "BAD": 0, "ZERO": 0}
    add = pl.greedy_share_topup(targets, cur, prices)
    assert set(add) <= {"A"}                       # only the priceable name traded
    assert "BAD" not in add and "ZERO" not in add


def test_topup_noop_when_at_or_over_target():
    targets = {"A": 1000.0, "B": 1000.0}
    prices = pd.Series({"A": 100.0, "B": 100.0})
    cur = {"A": 10, "B": 12}                        # exactly at / above target
    assert pl.greedy_share_topup(targets, cur, prices) == {}


def test_topup_empty_and_nonpositive_budget():
    prices = pd.Series({"A": 100.0})
    assert pl.greedy_share_topup({}, {}, prices) == {}
    assert pl.greedy_share_topup({"A": 1000.0}, {"A": 0}, prices, budget=0) == {}
    assert pl.greedy_share_topup({"A": 1000.0}, {"A": 0}, prices, budget=-50) == {}


def test_topup_prioritises_most_underweight_first():
    # with a budget for only ONE share, it must go to the MORE underweight name
    targets = {"A": 1000.0, "B": 1000.0}
    prices = pd.Series({"A": 100.0, "B": 100.0})
    cur = {"A": 2, "B": 8}                          # A gap $800 >> B gap $200
    add = pl.greedy_share_topup(targets, cur, prices, budget=100.0)
    assert add == {"A": 1}


def test_topup_lifts_book_from_floor_toward_target():
    # end-to-end: floor baseline is well under target; top-up closes most of it.
    targets = {t: 950.0 for t in ["A", "B", "C", "D", "E"]}
    prices = pd.Series({"A": 90.0, "B": 300.0, "C": 520.0, "D": 860.0, "E": 1200.0})
    floor = _floor_shares(targets, prices)         # E -> 0 (dropped), others under
    floor_val = sum(floor[t] * prices[t] for t in targets)
    add = pl.greedy_share_topup(targets, floor, prices)
    new_val = sum((floor[t] + add.get(t, 0)) * prices[t] for t in targets)
    target_val = sum(targets.values())
    assert new_val > floor_val                     # strictly more deployed
    # closes the gap to within one worst-case share of target
    assert target_val - new_val <= max(prices) + 1e-6


# --- merge_share_orders ----------------------------------------------------
def test_merge_folds_extra_into_existing_ticker():
    orders = [{"ticker": "A", "shares": 3, "action": "BUY", "price": 10.0}]
    merged = pl.merge_share_orders(orders, {"A": 2}, pd.Series({"A": 10.0}))
    assert len(merged) == 1 and merged[0]["shares"] == 5 and merged[0]["action"] == "BUY"


def test_merge_adds_new_ticker_with_price():
    orders = [{"ticker": "A", "shares": 3, "action": "BUY", "price": 10.0}]
    merged = pl.merge_share_orders(orders, {"B": 4}, pd.Series({"A": 10.0, "B": 25.0}))
    b = [o for o in merged if o["ticker"] == "B"][0]
    assert b["shares"] == 4 and b["action"] == "BUY" and b["price"] == pytest.approx(25.0)


def test_merge_drops_orders_that_net_to_zero():
    orders = [{"ticker": "A", "shares": -2, "action": "SELL", "price": 10.0}]
    merged = pl.merge_share_orders(orders, {"A": 2}, pd.Series({"A": 10.0}))
    assert merged == []                            # -2 + 2 = 0 -> removed


def test_merge_preserves_untouched_orders():
    orders = [{"ticker": "A", "shares": 1, "action": "BUY", "price": 10.0},
              {"ticker": "B", "shares": 2, "action": "BUY", "price": 20.0}]
    merged = pl.merge_share_orders(orders, {"A": 1}, pd.Series({"A": 10.0, "B": 20.0}))
    tickers = {o["ticker"]: o["shares"] for o in merged}
    assert tickers == {"A": 2, "B": 2}


# --- sleeve_breakdown ------------------------------------------------------
def test_sleeve_breakdown_percentages_and_cash():
    b = pl.sleeve_breakdown(nav=50_000, equity_val=28_500, bond_val=5_000,
                            fx_gross=13_000)
    assert b["equity"]["pct"] == pytest.approx(0.57)
    assert b["bonds"]["pct"] == pytest.approx(0.10)
    # FX is dollar-neutral -> cash is the remainder (NAV - equity - bonds)
    assert b["cash"]["value"] == pytest.approx(50_000 - 28_500 - 5_000)
    assert b["equity"]["value"] + b["bonds"]["value"] + b["cash"]["value"] \
        == pytest.approx(50_000)


def test_sleeve_breakdown_flags_equity_majority():
    good = pl.sleeve_breakdown(50_000, 28_500, 5_000, 13_000)
    assert good["equity_is_largest"] and good["equity_over_half"]
    # the pre-fix book: equity 43.6%, cash the biggest sleeve -> NOT majority
    bad = pl.sleeve_breakdown(50_000, 21_800, 5_000, 13_000)
    assert not bad["equity_is_largest"] and not bad["equity_over_half"]


def test_sleeve_breakdown_derisked_regime_is_not_a_failure():
    # heavy vol-target de-risk: equity legitimately below cash, flag reflects it
    b = pl.sleeve_breakdown(50_000, 16_000, 5_000, 13_000)
    assert not b["equity_is_largest"]              # cash 29k > equity 16k
    assert b["cash"]["value"] == pytest.approx(29_000)
