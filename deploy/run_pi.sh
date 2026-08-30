#!/bin/bash
# Scheduled STOCK + BOND rebalance (weekly exposure check / quarterly holdings).
# Self-guards the gateway before trading; auto_rebalance additionally refuses to
# route orders against an untrusted feed (check_tradeable / TradingHalt).
cd /home/maxlovinger/autoPortfolio || exit 1
source .venv/bin/activate

echo "=== $(date) : auto_rebalance run ===" >> auto_run.log
if ! bash ensure_gateway.sh >> auto_run.log 2>&1; then
  echo "$(date): ABORT auto_rebalance — gateway not healthy after restarts; nothing traded." >> auto_run.log
  exit 1
fi
python3 auto_rebalance.py --ibkr --gateway --live >> auto_run.log 2>&1
