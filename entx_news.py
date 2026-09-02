"""
entx_news.py — standalone news reporter for a single ticker (default ENTX,
Entera Bio). Fetches ticker-tagged headlines from Alpha Vantage's NEWS_SENTIMENT
feed, keeps only articles it hasn't sent before (dedupe state file), scores each
one two ways — Alpha Vantage's own per-ticker sentiment AND this repo's
news_sentiment scorer (FinBERT -> VADER -> bag-of-words) — and emails a compact
digest. If nothing is new, it sends nothing and exits 0.

Deliberately isolated from the trading system:
  * NO IBKR connection — it can't touch the Gateway session or the live book.
  * Own log (entx_news.log), own state (entx_<ticker>_seen.json), own cron line.
  * Reuses only the shared .env (Alpha Vantage key + the SMTP_* creds the daily
    snapshot already uses); imports nothing that opens a broker socket.

FinBERT is loaded LAZILY (only when there are genuinely new articles) so the
frequent "nothing new" runs cost just one HTTP call.

Usage:
  python3 entx_news.py [TICKER] [--send] [--all] [--min-rel 0.1]
    (default)  print the digest to stdout (nothing emailed) — for testing
    --send     email the digest via SMTP (only if there are new articles)
    --all      ignore the seen-state and show the most recent articles (test /
               first look); does NOT mark them seen unless combined with --send
    --min-rel  drop articles whose ticker relevance is below this (default 0.10)
"""
from __future__ import annotations
import json
import os
import ssl
import sys
import smtplib
import urllib.parse
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
AV_URL = "https://www.alphavantage.co/query"
SEEN_KEEP = 800                      # cap the dedupe file so it can't grow forever
DEFAULT_MIN_REL = 0.10               # ticker relevance floor (AV relevance in [0,1])


# --------------------------------------------------------------------- env
def load_env():
    """Populate os.environ from the repo .env (same file the snapshot reads)."""
    envp = HERE / ".env"
    if envp.exists():
        for line in envp.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def log(msg):
    line = f"{pd.Timestamp.now(tz='UTC').isoformat(timespec='seconds')}  {msg}"
    with open(HERE / "entx_news.log", "a") as f:
        f.write(line + "\n")
    print(line)


# --------------------------------------------------------------------- fetch
def fetch_feed(ticker: str, limit: int = 50) -> list[dict]:
    """Raw Alpha Vantage NEWS_SENTIMENT feed for `ticker` (list of article dicts)."""
    key = os.getenv("ALPHA_VANTAGE_API")
    if not key:
        raise RuntimeError("ALPHA_VANTAGE_API not set in .env")
    q = urllib.parse.urlencode({
        "function": "NEWS_SENTIMENT", "tickers": ticker,
        "sort": "LATEST", "limit": str(limit), "apikey": key,
    })
    with urllib.request.urlopen(f"{AV_URL}?{q}", timeout=30) as r:
        data = json.load(r)
    # AV returns {"Information": ...} on rate-limit / bad key instead of a feed.
    if "feed" not in data:
        note = data.get("Information") or data.get("Note") or data.get("Error Message") or data
        raise RuntimeError(f"Alpha Vantage returned no feed: {note}")
    return data["feed"]


# ----------------------------------------------------------------- pure core
def parse_feed(raw: list[dict], ticker: str, min_rel: float = DEFAULT_MIN_REL) -> list[dict]:
    """Normalize AV items to our article dicts, keeping only ones tagged for
    `ticker` above the relevance floor. Pure (no network) so it's unit-testable."""
    out = []
    for it in raw:
        ts = next((t for t in it.get("ticker_sentiment", [])
                   if t.get("ticker") == ticker), None)
        if ts is None:
            continue
        try:
            rel = float(ts.get("relevance_score", 0.0))
            av_sent = float(ts.get("ticker_sentiment_score", 0.0))
        except (TypeError, ValueError):
            continue
        if rel < min_rel:
            continue
        out.append({
            "url": it.get("url", ""),
            "title": (it.get("title") or "").strip(),
            "summary": (it.get("summary") or "").strip(),
            "source": it.get("source", "?"),
            "time": it.get("time_published", ""),      # e.g. 20260823T154054 (UTC)
            "relevance": rel,
            "av_sent": av_sent,
            "av_label": ts.get("ticker_sentiment_label", "?"),
        })
    # highest relevance first
    out.sort(key=lambda a: a["relevance"], reverse=True)
    return out


def load_seen(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_seen(path: Path, seen: dict):
    # keep only the most recently added SEEN_KEEP entries (insertion order)
    if len(seen) > SEEN_KEEP:
        seen = dict(list(seen.items())[-SEEN_KEEP:])
    path.write_text(json.dumps(seen))


def filter_new(articles: list[dict], seen: dict) -> list[dict]:
    """Articles whose url we haven't recorded before."""
    return [a for a in articles if a["url"] and a["url"] not in seen]


def _fmt_et(av_time: str) -> str:
    """AV timestamp (UTC, YYYYMMDDTHHMMSS) -> 'Aug 23, 11:40am ET'."""
    try:
        t = pd.to_datetime(av_time, format="%Y%m%dT%H%M%S", utc=True)
        return t.tz_convert("America/New_York").strftime("%b %d, %-I:%M%p ET")
    except Exception:
        return av_time or "?"


def score_finbert(articles: list[dict]) -> tuple[list[float], str]:
    """Score each article's headline with this repo's news_sentiment scorer.
    Returns (scores, backend_name). Loaded lazily; degrades gracefully if the
    module/model is unavailable (then scores are all None and backend='none')."""
    if not articles:
        return [], "none"
    try:
        import news_sentiment as ns
        backend = ns.available_backend(prefer="finbert")
        scores = list(ns.score_texts([a["title"] for a in articles], prefer="finbert"))
        return [float(s) for s in scores], backend
    except Exception as e:                       # never let scoring break the report
        log(f"note: local sentiment scorer unavailable ({e}); showing AV score only")
        return [None] * len(articles), "none"


def _label(x: float) -> str:
    if x is None:
        return "n/a"
    if x >= 0.15:
        return "Bullish"
    if x <= -0.15:
        return "Bearish"
    return "Neutral"


# ------------------------------------------------------------------ render
def build_digest(ticker: str, new: list[dict], fb_scores: list, backend: str):
    """(subject, html, text) for the new-article digest. Assumes new is non-empty."""
    n = len(new)
    rel_sum = sum(a["relevance"] for a in new) or 1.0
    av_net = sum(a["av_sent"] * a["relevance"] for a in new) / rel_sum   # rel-weighted
    fb_pairs = [(s, a["relevance"]) for s, a in zip(fb_scores, new) if s is not None]
    fb_net = (sum(s * r for s, r in fb_pairs) / (sum(r for _, r in fb_pairs) or 1.0)
              if fb_pairs else None)
    ts = pd.Timestamp.now(tz="America/New_York").strftime("%a %b %d, %-I:%M %p ET")

    def col(x):
        if x is None:
            return "#888"
        return "#137333" if x > 0.05 else ("#a50e0e" if x < -0.05 else "#555")

    rows = []
    txt_lines = []
    for a, fb in zip(new, fb_scores):
        fb_txt = f"{fb:+.2f}" if fb is not None else "n/a"
        rows.append(f"""<tr>
  <td style="padding:8px 10px;vertical-align:top">
    <a href="{a['url']}" style="color:#1a0dab;text-decoration:none;font-weight:600">{a['title']}</a>
    <div style="color:#888;font-size:12px;margin-top:2px">{_fmt_et(a['time'])} · {a['source']} · rel {a['relevance']*100:.0f}%</div>
  </td>
  <td style="padding:8px 10px;text-align:right;white-space:nowrap;color:{col(a['av_sent'])}">
    <b>{a['av_sent']:+.2f}</b><div style="font-size:11px;color:#999">{a['av_label']}</div></td>
  <td style="padding:8px 10px;text-align:right;white-space:nowrap;color:{col(fb)}">
    <b>{fb_txt}</b><div style="font-size:11px;color:#999">{_label(fb)}</div></td>
</tr>""")
        txt_lines.append(f"- {_fmt_et(a['time'])} | {a['title']}\n"
                         f"    {a['source']} · rel {a['relevance']*100:.0f}% · "
                         f"AV {a['av_sent']:+.2f} ({a['av_label']}) · "
                         f"{backend} {fb_txt}\n    {a['url']}")

    fb_net_txt = f"{fb_net:+.2f}" if fb_net is not None else "n/a"
    html = f"""<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:620px;margin:auto">
<h2 style="margin:0 0 2px">{ticker} — {n} new article{'s' if n!=1 else ''}</h2>
<div style="color:#666;font-size:13px;margin-bottom:12px">{ts}</div>
<div style="font-size:14px;margin-bottom:12px">
  Net sentiment (relevance-weighted): AlphaVantage <b style="color:{col(av_net)}">{av_net:+.2f}</b>
  · {backend} <b style="color:{col(fb_net)}">{fb_net_txt}</b></div>
<table style="border-collapse:collapse;width:100%;font-size:14px;border:1px solid #eee">
<tr style="background:#f5f6f8;font-weight:600">
  <td style="padding:6px 10px">Headline</td>
  <td style="padding:6px 10px;text-align:right">AV</td>
  <td style="padding:6px 10px;text-align:right">{backend}</td></tr>
{''.join(rows)}
</table>
<div style="color:#999;font-size:11px;margin-top:14px">Alpha Vantage NEWS_SENTIMENT · dual-scored (AV + {backend}) · new-articles-only digest.</div>
</div>"""

    text = (f"{ticker} — {n} new article{'s' if n!=1 else ''} — {ts}\n"
            f"Net sentiment (rel-weighted): AV {av_net:+.2f} · {backend} {fb_net_txt}\n\n"
            + "\n".join(txt_lines) + "\n")
    subj = f"{ticker} news: {n} new · AV {av_net:+.2f} · {backend} {fb_net_txt}"
    return subj, html, text


# ------------------------------------------------------------------- email
def send_email(subj, html, text) -> bool:
    host = os.getenv("SMTP_HOST"); port = int(os.getenv("SMTP_PORT", "465"))
    user = os.getenv("SMTP_USER"); pw = os.getenv("SMTP_PASS")
    to = os.getenv("ENTX_NEWS_TO") or os.getenv("SNAPSHOT_TO", user)
    if not (host and user and pw):
        log("SMTP env not set (SMTP_HOST/SMTP_USER/SMTP_PASS) — cannot send.")
        return False
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subj; msg["From"] = user; msg["To"] = to
    msg.attach(MIMEText(text, "plain")); msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context()) as s:
        s.login(user, pw); s.sendmail(user, [to], msg.as_string())
    log(f"{msg['Subject']} -> emailed to {to}")
    return True


# -------------------------------------------------------------------- main
def main():
    load_env()
    args = sys.argv[1:]
    send = "--send" in args
    show_all = "--all" in args
    min_rel = DEFAULT_MIN_REL
    positional = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--min-rel":
            if i + 1 < len(args):
                try: min_rel = float(args[i + 1])
                except ValueError: pass
                i += 1
        elif not a.startswith("--"):
            positional.append(a)
        i += 1
    ticker = (positional[0] if positional else os.getenv("ENTX_TICKER", "ENTX")).upper()

    seen_path = HERE / f"entx_{ticker.lower()}_seen.json"
    try:
        raw = fetch_feed(ticker)
    except Exception as e:
        log(f"ABORT: {e}")
        return
    articles = parse_feed(raw, ticker, min_rel=min_rel)
    seen = load_seen(seen_path)
    new = articles if show_all else filter_new(articles, seen)

    if not new:
        log(f"{ticker}: no new articles ({len(articles)} in feed, all seen).")
        return

    fb_scores, backend = score_finbert(new)
    subj, html, text = build_digest(ticker, new, fb_scores, backend)

    if send:
        if send_email(subj, html, text):
            now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
            for a in new:
                seen[a["url"]] = now
            save_seen(seen_path, seen)
            log(f"{ticker}: emailed {len(new)} new; seen file now {len(seen)} urls.")
    else:
        print("SUBJECT:", subj)
        print(text)
        print("(dry-run — nothing emailed, seen-state unchanged; use --send)")


if __name__ == "__main__":
    main()
