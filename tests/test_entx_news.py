"""Tests for entx_news.py pure core — parsing the Alpha Vantage feed, the
relevance floor, dedupe (only-if-new), seen-file capping, and digest rendering.
No network: feeds are hand-built dicts mirroring AV's NEWS_SENTIMENT shape."""
import json
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import entx_news as en


class _Resp:
    """Minimal stand-in for an HTTP response context manager (json.load reads it)."""
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()
    def read(self, *a):
        return self._b
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def _item(url, title, ticker="ENTX", rel=1.0, sent=0.4, label="Bullish", other=None):
    ts = [{"ticker": ticker, "relevance_score": str(rel),
           "ticker_sentiment_score": str(sent), "ticker_sentiment_label": label}]
    if other:                                    # a second, unrelated ticker tag
        ts.append({"ticker": other, "relevance_score": "0.9",
                   "ticker_sentiment_score": "0.9", "ticker_sentiment_label": "Bullish"})
    return {"url": url, "title": title, "summary": "", "source": "TestWire",
            "time_published": "20260823T154054", "ticker_sentiment": ts}


# ------------------------------------------------------------------ parse
def test_parse_keeps_only_target_ticker_and_reads_scores():
    raw = [_item("u1", "Entera up", rel=0.8, sent=0.42, label="Bullish")]
    arts = en.parse_feed(raw, "ENTX")
    assert len(arts) == 1
    a = arts[0]
    assert a["url"] == "u1" and a["relevance"] == 0.8
    assert a["av_sent"] == 0.42 and a["av_label"] == "Bullish"


def test_parse_drops_articles_not_tagged_for_ticker():
    raw = [_item("u1", "About AAPL only", ticker="AAPL")]
    assert en.parse_feed(raw, "ENTX") == []


def test_parse_default_keeps_all_relevance():
    # default floor is 0.0 — even a barely-relevant mention is reported
    raw = [_item("hi", "relevant", rel=0.5), _item("lo", "noise", rel=0.02)]
    arts = en.parse_feed(raw, "ENTX")             # DEFAULT_MIN_REL == 0.0
    assert {a["url"] for a in arts} == {"hi", "lo"}


def test_parse_relevance_floor_optional():
    # a caller can still opt into a floor via --min-rel
    raw = [_item("hi", "relevant", rel=0.5), _item("lo", "noise", rel=0.02)]
    arts = en.parse_feed(raw, "ENTX", min_rel=0.10)
    assert [a["url"] for a in arts] == ["hi"]


def test_parse_sorts_by_relevance_desc():
    raw = [_item("mid", "b", rel=0.5), _item("top", "a", rel=0.9), _item("low", "c", rel=0.2)]
    arts = en.parse_feed(raw, "ENTX")
    assert [a["url"] for a in arts] == ["top", "mid", "low"]


def test_parse_reads_ticker_tag_among_several():
    raw = [_item("u1", "multi", rel=0.7, sent=0.3, other="MSFT")]
    arts = en.parse_feed(raw, "ENTX")
    assert len(arts) == 1 and arts[0]["av_sent"] == 0.3   # ENTX tag, not MSFT's 0.9


# ------------------------------------------------------------------ dedupe
def test_filter_new_excludes_seen_urls():
    arts = en.parse_feed([_item("u1", "a"), _item("u2", "b")], "ENTX")
    new = en.filter_new(arts, {"u1": "2026-08-23T00:00:00+00:00"})
    assert [a["url"] for a in new] == ["u2"]


def test_filter_new_all_seen_returns_empty():
    arts = en.parse_feed([_item("u1", "a")], "ENTX")
    assert en.filter_new(arts, {"u1": "t"}) == []


def test_seen_roundtrip_and_cap(tmp_path):
    p = tmp_path / "seen.json"
    seen = {f"u{i}": "t" for i in range(en.SEEN_KEEP + 50)}
    en.save_seen(p, seen)
    back = en.load_seen(p)
    assert len(back) == en.SEEN_KEEP               # capped
    assert f"u{en.SEEN_KEEP + 49}" in back         # keeps the newest, drops oldest
    assert "u0" not in back


def test_load_seen_missing_file_is_empty(tmp_path):
    assert en.load_seen(tmp_path / "nope.json") == {}


def test_load_seen_corrupt_file_is_empty(tmp_path):
    p = tmp_path / "seen.json"; p.write_text("{ not valid json")
    assert en.load_seen(p) == {}                          # corrupt state -> empty, no crash


# ------------------------------------------------------------------ render
def test_build_digest_dual_scores_and_net():
    new = en.parse_feed([_item("u1", "Big win", rel=1.0, sent=0.4, label="Bullish"),
                         _item("u2", "Setback", rel=0.5, sent=-0.3, label="Bearish")], "ENTX")
    fb = [0.5, -0.2]
    subj, html, text = en.build_digest("ENTX", new, fb, "finbert")
    # both articles present
    assert "Big win" in html and "Setback" in html
    # AV net is relevance-weighted: (0.4*1.0 + -0.3*0.5)/1.5 = +0.1667
    assert "AV +0.17" in subj
    # finbert net: (0.5*1.0 + -0.2*0.5)/1.5 = +0.2667
    assert "finbert +0.27" in subj
    # both score columns rendered in text
    assert "AV +0.40" in text and "finbert +0.50" in text


def test_build_digest_handles_missing_finbert():
    new = en.parse_feed([_item("u1", "News", rel=1.0, sent=0.2)], "ENTX")
    subj, html, text = en.build_digest("ENTX", new, [None], "none")
    assert "n/a" in subj and "n/a" in text        # no crash when scorer unavailable
    assert "AV +0.20" in text


def test_build_digest_singular_plural():
    one = en.parse_feed([_item("u1", "solo")], "ENTX")
    subj, _, _ = en.build_digest("ENTX", one, [0.1], "finbert")
    assert "1 new" in subj and "articles" not in subj


def test_build_digest_includes_price_when_quoted():
    new = en.parse_feed([_item("u1", "news")], "ENTX")
    q = {"price": 3.45, "change": 0.17, "change_pct": 5.18,
         "prev_close": 3.28, "day": "2026-08-23"}
    subj, html, text = en.build_digest("ENTX", new, [0.1], "finbert", quote=q)
    assert "$3.45" in html and "$3.45" in text and "$3.45" in subj
    assert "+5.18%" in html and "+5.2%" in subj    # up-day formatting
    assert "▲" in html


def test_build_digest_price_optional():
    new = en.parse_feed([_item("u1", "news")], "ENTX")
    subj, html, text = en.build_digest("ENTX", new, [0.1], "finbert", quote=None)
    assert "Price" not in html and "Price" not in text   # no price line, no crash


# --------------------------------------------------------- small formatters
def test_fmt_et_valid_and_fallback():
    assert "Aug 23" in en._fmt_et("20260823T154054")     # UTC -> ET, formatted
    assert en._fmt_et("not-a-time") == "not-a-time"       # unparseable -> passthrough


def test_label_thresholds():
    assert en._label(0.2) == "Bullish"
    assert en._label(-0.2) == "Bearish"
    assert en._label(0.0) == "Neutral"
    assert en._label(None) == "n/a"


def test_price_bits_down_day_and_none():
    h, t = en._price_bits({"price": 2.90, "change": -0.05, "change_pct": -1.69,
                           "day": "2026-09-01"})
    assert "▼" in h and "#a50e0e" in h and "$2.90" in h   # down-day = red arrow
    assert "$2.90" in t and "-1.69%" in t
    assert en._price_bits(None) == ("", "")               # no quote -> empty


# ------------------------------------------------------- parse edge cases
def test_parse_skips_item_without_ticker_sentiment():
    assert en.parse_feed([{"url": "u1", "title": "x"}], "ENTX") == []


def test_parse_skips_malformed_scores():
    raw = [{"url": "u1", "title": "x", "source": "s", "time_published": "t",
            "ticker_sentiment": [{"ticker": "ENTX", "relevance_score": "oops",
                                  "ticker_sentiment_score": "bad"}]}]
    assert en.parse_feed(raw, "ENTX") == []               # unparseable scores skipped


# ------------------------------------------------------------ env / log
def test_load_env_sets_from_dotenv(monkeypatch, tmp_path):
    monkeypatch.setattr(en, "HERE", tmp_path)
    (tmp_path / ".env").write_text('ENTX_TEST_KEY="hello"\n# a comment\n\n')
    monkeypatch.delenv("ENTX_TEST_KEY", raising=False)
    en.load_env()
    assert os.environ["ENTX_TEST_KEY"] == "hello"


def test_log_writes_file_and_prints(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(en, "HERE", tmp_path)
    en.log("hello-log")
    assert "hello-log" in (tmp_path / "entx_news.log").read_text()
    assert "hello-log" in capsys.readouterr().out


# ------------------------------------------------------ fetch_feed (mocked)
def test_fetch_feed_happy(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API", "k")
    monkeypatch.setattr(en.urllib.request, "urlopen",
                        lambda url, timeout=30: _Resp({"feed": [_item("u1", "A")]}))
    assert en.fetch_feed("ENTX")[0]["url"] == "u1"


def test_fetch_feed_rate_limited_raises(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API", "k")
    monkeypatch.setattr(en.urllib.request, "urlopen",
                        lambda url, timeout=30: _Resp({"Information": "rate limit hit"}))
    with pytest.raises(RuntimeError, match="rate limit"):
        en.fetch_feed("ENTX")


def test_fetch_feed_no_key_raises(monkeypatch):
    monkeypatch.delenv("ALPHA_VANTAGE_API", raising=False)
    with pytest.raises(RuntimeError, match="ALPHA_VANTAGE_API"):
        en.fetch_feed("ENTX")


# ----------------------------------------------------- fetch_quote (mocked)
def test_fetch_quote_happy(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API", "k")
    payload = {"Global Quote": {"05. price": "3.45", "09. change": "0.17",
                                "10. change percent": "5.18%", "08. previous close": "3.28",
                                "07. latest trading day": "2026-08-23"}}
    monkeypatch.setattr(en.urllib.request, "urlopen", lambda url, timeout=20: _Resp(payload))
    q = en.fetch_quote("ENTX")
    assert q["price"] == 3.45 and q["change_pct"] == 5.18 and q["day"] == "2026-08-23"


def test_fetch_quote_empty_returns_none(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API", "k")
    monkeypatch.setattr(en.urllib.request, "urlopen",
                        lambda url, timeout=20: _Resp({"Global Quote": {}}))
    assert en.fetch_quote("ENTX") is None


def test_fetch_quote_exception_returns_none(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API", "k")
    def boom(*a, **k): raise OSError("net down")
    monkeypatch.setattr(en.urllib.request, "urlopen", boom)
    assert en.fetch_quote("ENTX") is None                 # best-effort, never raises


def test_fetch_quote_no_key_returns_none(monkeypatch):
    monkeypatch.delenv("ALPHA_VANTAGE_API", raising=False)
    assert en.fetch_quote("ENTX") is None


# ------------------------------------------------- score_finbert (mocked)
def test_score_finbert_empty_is_none_backend():
    assert en.score_finbert([]) == ([], "none")


def test_score_finbert_uses_scorer(monkeypatch):
    fake = types.SimpleNamespace(
        available_backend=lambda prefer="finbert": "finbert",
        score_texts=lambda texts, prefer="finbert": [0.5] * len(texts))
    monkeypatch.setitem(sys.modules, "news_sentiment", fake)
    scores, backend = en.score_finbert([{"title": "a"}, {"title": "b"}])
    assert scores == [0.5, 0.5] and backend == "finbert"


def test_score_finbert_graceful_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(en, "HERE", tmp_path)             # log() writes here, not the repo
    def boom(*a, **k): raise RuntimeError("model missing")
    fake = types.SimpleNamespace(
        available_backend=lambda prefer="finbert": "finbert", score_texts=boom)
    monkeypatch.setitem(sys.modules, "news_sentiment", fake)
    scores, backend = en.score_finbert([{"title": "a"}])
    assert scores == [None] and backend == "none"         # degrades, doesn't crash


# ------------------------------------------------------ send_email (mocked)
def test_send_email_no_env_returns_false(monkeypatch, tmp_path):
    monkeypatch.setattr(en, "HERE", tmp_path)
    for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS"):
        monkeypatch.delenv(k, raising=False)
    assert en.send_email("s", "<b>h</b>", "t") is False


def test_send_email_sends_and_prefers_entx_recipient(monkeypatch, tmp_path):
    monkeypatch.setattr(en, "HERE", tmp_path)
    monkeypatch.setenv("SMTP_HOST", "smtp.x"); monkeypatch.setenv("SMTP_PORT", "465")
    monkeypatch.setenv("SMTP_USER", "me@x.com"); monkeypatch.setenv("SMTP_PASS", "pw")
    monkeypatch.setenv("SNAPSHOT_TO", "snap@x.com"); monkeypatch.setenv("ENTX_NEWS_TO", "news@x.com")
    rec = {}
    class FakeSMTP:
        def __init__(self, host, port, context=None): rec.update(host=host, port=port)
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def login(self, u, p): rec["login"] = (u, p)
        def sendmail(self, frm, to, msg): rec.update(to=to, frm=frm)
    monkeypatch.setattr(en.smtplib, "SMTP_SSL", FakeSMTP)
    assert en.send_email("subj", "<b>h</b>", "t") is True
    assert rec["to"] == ["news@x.com"]                    # ENTX_NEWS_TO beats SNAPSHOT_TO
    assert rec["host"] == "smtp.x" and rec["port"] == 465
    assert rec["login"] == ("me@x.com", "pw")


# ------------------------------------------------------------- main (mocked)
def _wire_main(monkeypatch, tmp_path, raw, quote=None):
    monkeypatch.setattr(en, "HERE", tmp_path)
    monkeypatch.setattr(en, "fetch_feed", lambda t, limit=50: raw)
    monkeypatch.setattr(en, "fetch_quote", lambda t: quote)
    monkeypatch.setattr(en, "score_finbert", lambda new: ([0.1] * len(new), "finbert"))


def test_main_send_writes_seen_then_noop(monkeypatch, tmp_path):
    _wire_main(monkeypatch, tmp_path, [_item("u1", "A"), _item("u2", "B")])
    sent = []
    monkeypatch.setattr(en, "send_email", lambda s, h, t: sent.append(s) or True)
    monkeypatch.setattr(sys, "argv", ["entx_news.py", "ENTX", "--send"])
    en.main()
    assert len(sent) == 1                                 # emailed the 2 new articles
    seen = en.load_seen(tmp_path / "entx_entx_seen.json")
    assert set(seen) == {"u1", "u2"}                      # both recorded
    # second run: same feed, everything seen -> no email
    sent.clear()
    en.main()
    assert sent == []


def test_main_dryrun_prints_and_leaves_no_state(monkeypatch, tmp_path, capsys):
    _wire_main(monkeypatch, tmp_path, [_item("u1", "A")])
    called = []
    monkeypatch.setattr(en, "send_email", lambda *a: called.append(1) or True)
    monkeypatch.setattr(sys, "argv", ["entx_news.py", "ENTX"])   # no --send
    en.main()
    out = capsys.readouterr().out
    assert "SUBJECT:" in out and "dry-run" in out
    assert called == []                                   # nothing emailed
    assert not (tmp_path / "entx_entx_seen.json").exists()  # dry-run doesn't touch state


def test_main_aborts_on_fetch_error(monkeypatch, tmp_path):
    monkeypatch.setattr(en, "HERE", tmp_path)
    def boom(*a, **k): raise RuntimeError("rate limit")
    monkeypatch.setattr(en, "fetch_feed", boom)
    called = []
    monkeypatch.setattr(en, "send_email", lambda *a: called.append(1))
    monkeypatch.setattr(sys, "argv", ["entx_news.py", "ENTX", "--send"])
    en.main()                                             # must not raise
    assert called == []


def test_main_min_rel_flag_filters(monkeypatch, tmp_path, capsys):
    _wire_main(monkeypatch, tmp_path,
               [_item("hi", "A", rel=0.9), _item("lo", "B", rel=0.02)])
    monkeypatch.setattr(sys, "argv", ["entx_news.py", "ENTX", "--all", "--min-rel", "0.5"])
    en.main()
    out = capsys.readouterr().out
    assert "1 new" in out                                 # only the high-relevance one


def test_main_min_rel_nonnumeric_ignored(monkeypatch, tmp_path, capsys):
    _wire_main(monkeypatch, tmp_path, [_item("u1", "A", rel=0.9)])
    monkeypatch.setattr(sys, "argv", ["entx_news.py", "ENTX", "--all", "--min-rel", "abc"])
    en.main()                                             # bad value -> default 0.0, no crash
    assert "1 new" in capsys.readouterr().out
