#!/bin/bash
# Scheduled FX carry sleeve rebalance. Self-guards the gateway before trading;
# run_fx additionally aborts on a $0 NAV or a hard connectivity-loss code seen
# during the account/position download.
cd /home/maxlovinger/autoPortfolio || exit 1
source .venv/bin/activate

echo "=== $(date) : run_fx (monthly carry rebalance) ===" >> fx_run.log
if ! bash ensure_gateway.sh >> fx_run.log 2>&1; then
  echo "$(date): ABORT run_fx — gateway not healthy after restarts; nothing traded." >> fx_run.log
  exit 1
fi
python3 run_fx.py --live --gateway --nlong 1 --nshort 1 >> fx_run.log 2>&1
