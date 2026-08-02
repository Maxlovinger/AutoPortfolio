#!/bin/zsh
# Wrapper for the scheduled auto-rebalance job (called by launchd).
# Runs in the project dir so relative files (universe.csv, state, logs) resolve.
#
# SAFE DEFAULT: SimBroker (self-contained paper account) — no external orders,
# just builds the honest theoretical track record + decision_log.md each run.
#
# TO GO LIVE on the IBKR paper account (requires TWS running with API enabled):
#   change the line below to:
#     "$PY" auto_rebalance.py --ibkr --live >> auto_run.log 2>&1
#
cd /Users/max_lovinger/Documents/autoPortfolio || exit 1
PY=/Users/max_lovinger/anaconda3/bin/python3
echo "=== $(date) : auto_rebalance run ===" >> auto_run.log
# LIVE on the IBKR paper account (requires TWS running during market hours).
# To revert to the safe self-contained track, drop "--ibkr --live".
"$PY" auto_rebalance.py --ibkr --live >> auto_run.log 2>&1
