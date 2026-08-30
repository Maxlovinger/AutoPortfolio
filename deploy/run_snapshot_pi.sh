#!/bin/bash
cd /home/maxlovinger/autoPortfolio || exit 1
source .venv/bin/activate
python3 portfolio_snapshot.py --send >> snapshot.log 2>&1
