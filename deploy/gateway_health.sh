#!/bin/bash
# Standalone gateway health monitor (cron: intraday). Restarts the container if
# the Gateway isn't logged in. The trade scripts also call ensure_gateway.sh at
# trade time; this is the general daytime auto-heal / early warm-up.
cd /home/maxlovinger/autoPortfolio || exit 1
source .venv/bin/activate
if python3 gateway_health.py >> health.log 2>&1; then
  echo "$(date): gateway healthy" >> health.log
else
  echo "$(date): gateway UNHEALTHY -> restarting container" >> health.log
  (cd /home/maxlovinger/ib-gateway && docker compose restart) >> /home/maxlovinger/autoPortfolio/health.log 2>&1
fi
