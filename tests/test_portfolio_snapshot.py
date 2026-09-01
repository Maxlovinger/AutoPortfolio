"""
Tests for the daily snapshot email's P&L resolution + rendering.

Regression target: the report always showed "Unrealized P&L $0" because it read
account key "UnrealizedPnL", but IB streams it as "$LEDGER-UnrealizedPnL". And it
had no daily P&L at all. resolve_pnl() now prefers reqPnL (daily + unrealized),
falling back to the ledger key, then per-position marks.
"""
import types
import portfolio_snapshot as ps


def _app(v=None, stk=None, fx=None, ledger=None, daily=None, unreal=None):
    a = types.SimpleNamespace()
    a.v = v or {"NetLiquidation": 248_000.0, "TotalCashValue": 73_000.0}
    a.stk = stk or {"AAPL": {"pos": 16, "mv": 3800.0, "upnl": -120.0},
                    "IEF": {"pos": 254, "mv": 24000.0, "upnl": -50.0}}
    a.fx = fx or []
    a.ledger = ledger or {}
    a.pnl_daily = daily
    a.pnl_unreal = unreal
    a.pnl_real = 0.0
    return a


# -------------------------------------------------------------------- resolve
def test_resolve_prefers_reqpnl():
    a = _app(daily=-844.24, unreal=-1788.98)
    daily, unreal = ps.resolve_pnl(a)
    assert daily == -844.24 and unreal == -1788.98


def test_resolve_falls_back_to_ledger_unrealized():
    # reqPnL didn't respond; the $LEDGER value was normalized into v["UnrealizedPnL"]
    a = _app(v={"NetLiquidation": 248_000.0, "TotalCashValue": 73_000.0,
               "UnrealizedPnL": -1743.33}, daily=None, unreal=None)
    daily, unreal = ps.resolve_pnl(a)
    assert unreal == -1743.33
    assert daily is None                       # honestly unknown -> not a fake 0


def test_resolve_falls_back_to_position_marks():
    a = _app(daily=None, unreal=None)          # no reqPnL, no ledger key
    daily, unreal = ps.resolve_pnl(a)
    assert unreal == -170.0                     # -120 + -50 summed from positions


def test_unrealized_is_never_silently_zero_when_data_exists():
    # the exact bug: data present but old code showed 0
    a = _app(v={"NetLiquidation": 248_000.0, "TotalCashValue": 73_000.0,
               "UnrealizedPnL": -1743.33}, daily=None, unreal=None)
    _, unreal = ps.resolve_pnl(a)
    assert unreal != 0.0


# -------------------------------------------------------------------- render
def test_build_renders_daily_and_unrealized():
    a = _app(daily=-844.24, unreal=-1788.98)
    subj, html, text = ps.build(a)
    # unrealized shown (not 0) in all three surfaces
    assert "1,789" in html and "1,789" in text
    assert "Unrl $-1,789" in subj
    # daily P&L present with sign + % in body, and in subject
    assert "Today's P&L" in html and "-844" in html and "%" in html
    assert "Today's P&L" in text and "-844" in text
    assert "Day $-844" in subj


def test_build_daily_na_when_unavailable():
    a = _app(daily=None, unreal=-1743.33)
    subj, html, text = ps.build(a)
    assert "n/a" in html and "Day n/a" in subj
    # unrealized still renders from the fallback
    assert "1,743" in html


# ------------------------------------------------------------------ FX sleeve
def test_fx_sleeve_read_from_ledger_not_positions():
    """Regression: FX read from reqPositions showed $0. It must come from the cash
    ledger — short CHF (~-$64.6k) + long ZAR (~+$64.7k), gross ~$129k."""
    a = _app(ledger={"CHF": {"cash": -52289.92, "rate": 1.2352072},
                     "ZAR": {"cash": 1042157.82, "rate": 0.0620553},
                     "USD": {"cash": 64604.0, "rate": 1.0},
                     "BASE": {"cash": 64687.0, "rate": 1.0},
                     "JPY": {"cash": 0.0, "rate": 0.0062}},   # dust -> excluded
             daily=-10.0, unreal=-1853.0)
    subj, html, text = ps.build(a)
    assert "FX gross $129," in text            # ~$129k gross, not $0
    assert "CHF" in html and "ZAR" in html     # both legs listed
    assert "USD" not in text.split("FX:")[1]   # base cash isn't an FX leg
    assert "JPY" not in html                    # dust filtered


def test_fx_sleeve_zero_when_flat():
    a = _app(ledger={"USD": {"cash": 100000.0, "rate": 1.0}}, daily=0.0, unreal=0.0)
    _, _, text = ps.build(a)
    assert "FX gross $0" in text
