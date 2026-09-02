# Raspberry Pi deployment (live IB Gateway)

These are the operational scripts that run the auto-portfolio live on the pi
(`ssh pi`). They live on the pi at `/home/maxlovinger/autoPortfolio/` alongside
the Python code; this folder is the version-controlled source of truth.

## Layout on the pi
- `/home/maxlovinger/autoPortfolio/` — Python code + these scripts + `.venv`
- `/home/maxlovinger/ib-gateway/` — `docker-compose.yml` for the `ib-gateway`
  container (IBC + IB Gateway 10.x). Maps host `127.0.0.1:4002` -> container API.

## Scripts
| Script | Role |
|---|---|
| `ensure_gateway.sh` | Verify gateway is logged in (NAV readable) **at trade time**; restart container + re-check up to 3x. Sourced by the trade jobs so they never trade against a broken feed. |
| `run_pi.sh` | Weekly/quarterly STOCK+BOND rebalance (`auto_rebalance.py --ibkr --gateway --live`). Guards via `ensure_gateway.sh`. |
| `run_fx_pi.sh` | FX carry sleeve (`run_fx.py --live --gateway`). Guards via `ensure_gateway.sh`. |
| `gateway_health.py` | Read-only probe: exit 0 if NAV readable, else 1. |
| `gateway_health.sh` | Intraday auto-heal / warm-up cron wrapper around the probe. |
| `run_snapshot_pi.sh` | Daily NAV snapshot email. |
| `run_entx_news_pi.sh` | ENTX (Entera Bio) news reporter. **No IBKR connection** — fully isolated from the trading system; emails only when a new article appears. |

## Safety model (two independent layers)
1. **Operational** — `ensure_gateway.sh` confirms the gateway is logged in
   immediately before each trade job (closes the gap between the 09:45 health
   cron and the 10:00 trade).
2. **In-code** — `auto_rebalance.check_tradeable()` (raises `TradingHalt`) and
   `run_fx` abort if NAV is `<= 0`, positions come back phantom-empty, or a hard
   connectivity-loss code (1100/2110) is seen. Nothing is transmitted on a trip.

## Crontab (unchanged times; scripts now self-guard)
```
0 10 * * 1-5   /home/maxlovinger/autoPortfolio/run_pi.sh          # rebalance 10:00 ET
15 10 1-7 * 1  /home/maxlovinger/autoPortfolio/run_fx_pi.sh       # FX carry (Mondays / month start)
5 17 * * *     /home/maxlovinger/autoPortfolio/run_snapshot_pi.sh # daily snapshot email
45 9 * * 1-5   /home/maxlovinger/autoPortfolio/gateway_health.sh  # warm-up
0 13 * * 1-5   /home/maxlovinger/autoPortfolio/gateway_health.sh  # intraday heal
0 9-16 * * 1-5 /home/maxlovinger/autoPortfolio/run_entx_news_pi.sh # ENTX news (hourly, market hrs; emails only if new)
```
> ENTX news cron: hourly on the top of the hour, 9am–4pm **pi local time**,
> weekdays. Confirm the pi's timezone (`timedatectl`) — if it's UTC not ET, shift
> the hours to cover the 09:30–16:00 ET session. The job is a safe no-op when
> there's nothing new, so a slightly wide window is harmless.

## Deploy
Copy changed files to the pi, then re-check perms:
```
scp deploy/*.sh deploy/gateway_health.py entx_news.py pi:/home/maxlovinger/autoPortfolio/
ssh pi 'chmod +x /home/maxlovinger/autoPortfolio/*.sh'
```
Then add the ENTX cron line above via `ssh pi 'crontab -e'`.

### ENTX reporter's isolated venv (`.venv-news`)
The news reporter runs under its **own** virtualenv, never the trading `.venv`,
because it needs torch + transformers (FinBERT) — which must not sit in the live
trading environment. Note the **CPU-only** torch index: the default `pip install
torch` pulls the CUDA build, whose CUDA-preload segfaults on the Pi's ARM.
```
cd /home/maxlovinger/autoPortfolio
python3 -m venv .venv-news && source .venv-news/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers pandas
```
