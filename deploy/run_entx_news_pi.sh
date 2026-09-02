#!/bin/bash
# ENTX (Entera Bio) news reporter. Fully isolated from the trading system — no
# IBKR connection, so it can't disturb the Gateway session or the live book.
# Emails only when there is a genuinely NEW article (dedupe state in
# entx_entx_seen.json); silent no-op otherwise. Safe to run frequently.
cd /home/maxlovinger/autoPortfolio || exit 1
source .venv/bin/activate
python3 entx_news.py ENTX --send >> entx_news.log 2>&1
