"""
Tests for the FRESH live carry source (fx/data.py BIS CBPOL policy rates). The
network fetch is mocked with a canned SDMX-CSV so these run offline; they pin
down: SDMX-CSV parsing + forward-fill (policy rates are step functions),
carry = foreign - USD with USD dropped, and full WIDE-universe area coverage.
"""
import pandas as pd
import pytest

import fx.data as fd


# a canned BIS CBPOL SDMX-CSV: three areas, KRW's last obs is OLDER than the
# others (it hasn't changed rates), which ffill must carry to the latest date.
CANNED = (
    "FREQ,REF_AREA,TITLE,TIME_PERIOD,OBS_VALUE,OBS_STATUS\n"
    "D,US,US policy rate,2026-08-04,3.625,A\n"
    "D,MX,MX policy rate,2026-08-04,6.500,A\n"
    "D,KR,KR policy rate,2026-07-01,2.500,A\n"
)
AREA_TO_CCY = {"US": "USD", "MX": "MXN", "KR": "KRW"}


def test_parse_cbpol_pivots_and_ffills():
    wide = fd.parse_cbpol(CANNED, AREA_TO_CCY)
    assert set(wide.columns) == {"USD", "MXN", "KRW"}
    # latest row is a COMPLETE cross-section: KRW's older 2.5 is carried forward
    last = wide.iloc[-1]
    assert last["USD"] == pytest.approx(3.625)
    assert last["MXN"] == pytest.approx(6.5)
    assert last["KRW"] == pytest.approx(2.5)          # ffilled from 2026-07-01
    assert str(wide.index[-1].date()) == "2026-08-04"


def test_parse_cbpol_ignores_unmapped_areas():
    extra = CANNED + "D,ZZ,junk,2026-08-04,9.9,A\n"
    wide = fd.parse_cbpol(extra, AREA_TO_CCY)
    assert "ZZ" not in wide.columns and set(wide.columns) == {"USD", "MXN", "KRW"}


def test_policy_carry_is_diff_vs_usd_usd_dropped():
    row = pd.Series({"USD": 3.625, "MXN": 6.5, "KRW": 2.5, "CHF": 0.0})
    carry = fd.policy_carry(row)
    assert "USD" not in carry.index
    assert carry["MXN"] == pytest.approx(2.875)       # 6.5 - 3.625
    assert carry["KRW"] == pytest.approx(-1.125)
    assert carry["CHF"] == pytest.approx(-3.625)      # funder = negative carry


def test_policy_area_covers_whole_wide_universe():
    # every tradeable currency in the live universe must map to a BIS area
    assert set(fd.POLICY_AREA) == set(fd.WIDE)
    assert fd.POLICY_AREA["EUR"] == "XM"              # euro area code


def test_fetch_policy_rates_uses_mock(monkeypatch):
    class _Resp:
        text = CANNED
        def raise_for_status(self): pass

    captured = {}

    def _fake_get(url, **kw):
        captured["url"] = url
        return _Resp()

    monkeypatch.setattr(fd.requests, "get", _fake_get)
    # restrict universe to the 3 canned currencies
    uni = {c: fd.WIDE[c] for c in ("USD", "MXN", "KRW")}
    wide = fd.fetch_policy_rates(uni)
    assert set(wide.columns) == {"USD", "MXN", "KRW"}
    # request built the SDMX multi-area key and asked for the latest obs
    assert "lastNObservations=1" in captured["url"]
    assert "US+MX+KR" in captured["url"] or "MX+US+KR" in captured["url"] \
        or "+".join(sorted(["US", "MX", "KR"])) in captured["url"] \
        or all(a in captured["url"] for a in ("US", "MX", "KR"))


def test_fresh_carry_end_to_end_mocked(monkeypatch):
    class _Resp:
        text = CANNED
        def raise_for_status(self): pass
    monkeypatch.setattr(fd.requests, "get", lambda url, **kw: _Resp())
    uni = {c: fd.WIDE[c] for c in ("USD", "MXN", "KRW")}
    carry = fd.fresh_carry(uni)
    assert "USD" not in carry.index
    assert carry["MXN"] == pytest.approx(2.875)
