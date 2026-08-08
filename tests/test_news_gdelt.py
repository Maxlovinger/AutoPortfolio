"""
Offline tests for news_gdelt.py. All HTTP is monkeypatched (GDELT is never
contacted); we test tone parsing/rescaling, monthly aggregation, the currency
mapping, caching, and the backoff-then-raise behavior.
"""
import json
import pandas as pd
import pytest

import news_gdelt as g


def _fake_timeline(dates_values):
    """Build a GDELT TimelineTone-shaped response dict."""
    return {"timeline": [{"data": [{"date": d, "value": v}
                                   for d, v in dates_values]}]}


# --- parsing / rescaling ---------------------------------------------------
def test_timeline_parse_rescale_and_clip():
    data = _fake_timeline([("2020-01-15T00:00:00Z", 5.0),
                           ("2020-02-15T00:00:00Z", -20.0)])  # out of range
    s = g._timeline_to_series(data)
    assert s.iloc[0] == pytest.approx(0.5)        # 5/10
    assert s.iloc[1] == pytest.approx(-1.0)       # clipped from -2.0
    assert s.index.is_monotonic_increasing


def test_empty_timeline_is_empty_series():
    assert g._timeline_to_series({}).empty
    assert g._timeline_to_series({"timeline": []}).empty


# --- aggregation -----------------------------------------------------------
def test_monthly_tone_resamples_to_month_end(monkeypatch):
    data = _fake_timeline([("2020-01-05T00:00:00Z", 2.0),
                           ("2020-01-20T00:00:00Z", 4.0),   # Jan mean -> 0.3
                           ("2020-02-10T00:00:00Z", -6.0)]) # Feb -> -0.6
    monkeypatch.setattr(g, "_fetch", lambda params, **k: data)
    m = g.monthly_tone("anything")
    assert len(m) == 2
    assert m.iloc[0] == pytest.approx(0.3)
    assert m.iloc[1] == pytest.approx(-0.6)


# --- currency mapping ------------------------------------------------------
def test_country_tone_uses_mapped_query(monkeypatch):
    seen = {}
    def fake_fetch(params, **k):
        seen["query"] = params["query"]
        return _fake_timeline([("2020-01-15T00:00:00Z", 1.0)])
    monkeypatch.setattr(g, "_fetch", fake_fetch)
    g.country_tone("JPY")
    assert "Japan" in seen["query"]


def test_country_tone_unknown_raises():
    with pytest.raises(KeyError):
        g.country_tone("XXX")


def test_entity_tone_quotes_name(monkeypatch):
    seen = {}
    def fake_fetch(params, **k):
        seen["query"] = params["query"]
        return _fake_timeline([("2020-01-15T00:00:00Z", 3.0)])
    monkeypatch.setattr(g, "_fetch", fake_fetch)
    g.entity_tone("Apple Inc")
    assert seen["query"] == '"Apple Inc"'


def test_country_tone_panel_columns(monkeypatch):
    monkeypatch.setattr(g, "_fetch", lambda params, **k:
                        _fake_timeline([("2020-01-15T00:00:00Z", 2.0),
                                        ("2020-02-15T00:00:00Z", 3.0)]))
    df = g.country_tone_panel(["EUR", "JPY", "AUD"])
    assert list(df.columns) == ["EUR", "JPY", "AUD"]
    assert len(df) == 2


# --- caching + backoff -----------------------------------------------------
def test_fetch_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "CACHE_DIR", str(tmp_path))
    calls = {"n": 0}
    payload = _fake_timeline([("2020-01-15T00:00:00Z", 1.0)])

    class FakeResp:
        def __init__(self, b): self.b = b
        def read(self): return self.b
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=30):
        calls["n"] += 1
        return FakeResp(json.dumps(payload).encode())

    monkeypatch.setattr(g.urllib.request, "urlopen", fake_urlopen)
    p = {"query": "x", "mode": "TimelineTone", "format": "json", "timespan": "24m"}
    g._fetch(p, throttle=0)         # miss -> hits network, writes cache
    g._fetch(p, throttle=0)         # hit  -> served from disk
    assert calls["n"] == 1


def test_fetch_raises_after_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(g.time, "sleep", lambda s: None)   # no real waiting
    def boom(req, timeout=30):
        raise OSError("429")
    monkeypatch.setattr(g.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="GDELT fetch failed"):
        g._fetch({"query": "x"}, retries=2, pause=0.0)
