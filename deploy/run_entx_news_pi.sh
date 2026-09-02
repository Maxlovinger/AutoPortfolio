#!/bin/bash
# ENTX (Entera Bio) news reporter. Fully isolated from the trading system — no
# IBKR connection, so it can't disturb the Gateway session or the live book.
# Emails only when there is a genuinely NEW article (dedupe state in
# entx_entx_seen.json); silent no-op otherwise. Safe to run frequently.
cd /home/maxlovinger/autoPortfolio || exit 1
# Uses its OWN venv (.venv-news), NOT the trading .venv — it carries torch +
# transformers (FinBERT) which have no business in the live trading environment.
# CPU-only torch (the default CUDA build segfaults on the Pi's ARM). Keeping the
# envs separate means a FinBERT/torch change can never disturb the book.
source .venv-news/bin/activate
python3 entx_news.py ENTX --send >> entx_news.log 2>&1
