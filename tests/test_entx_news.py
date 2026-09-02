"""Tests for entx_news.py pure core — parsing the Alpha Vantage feed, the
relevance floor, dedupe (only-if-new), seen-file capping, and digest rendering.
No network: feeds are hand-built dicts mirroring AV's NEWS_SENTIMENT shape."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import entx_news as en


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


def test_parse_applies_relevance_floor():
    raw = [_item("hi", "relevant", rel=0.5), _item("lo", "noise", rel=0.02)]
    arts = en.parse_feed(raw, "ENTX", min_rel=0.10)
    assert [a["url"] for a in arts] == ["hi"]     # low-relevance dropped


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
