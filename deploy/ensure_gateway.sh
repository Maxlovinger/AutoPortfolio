#!/bin/bash
# ensure_gateway.sh — verify the IB Gateway container is logged in (NAV readable)
# RIGHT BEFORE a scheduled trading job, restarting + re-checking if not.
#
# WHY: the standalone health cron runs at 09:45 but the rebalance runs at 10:00.
# If connectivity dropped in that window (or a 09:45 restart hadn't finished
# logging in by 10:00), the trade ran against a broken feed and read NAV $0.
# Calling this from the run scripts moves the check to trade time, so the job
# never trades against a down gateway. Belt-and-suspenders with the in-code
# check_tradeable() guard.
#
# Exit 0 = healthy (proceed to trade). Exit 1 = still down after 3 restarts (abort).
cd /home/maxlovinger/autoPortfolio || exit 1
source .venv/bin/activate

GW_DIR=/home/maxlovinger/ib-gateway

for attempt in 1 2 3; do
  if python3 gateway_health.py; then
    exit 0
  fi
  echo "$(date): gateway unhealthy (attempt ${attempt}/3) -> restarting container"
  (cd "$GW_DIR" && docker compose restart)
  sleep 90    # allow IBC to relaunch + log in before re-checking
done

exit 1
