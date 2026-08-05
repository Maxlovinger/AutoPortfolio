# Raspberry Pi Deployment Guide — autoPortfolio

Goal: run the automated strategy (`auto_rebalance.py`) unattended on an always-on
Raspberry Pi, so the weekly exposure check (Mondays) and quarterly holdings
rebalance fire reliably without a laptop being awake.

This is the standard hobbyist setup for IBKR automation and it removes every
problem the Mac had: no `~/Documents` TCC/permission failures, no laptop sleep,
no "did I leave TWS open," no daily manual re-login.

---

## 0. The architecture (what runs where)

Three things must be alive on the Pi at the scheduled moment:

1. **IB Gateway** — the headless, lightweight version of TWS. It holds the API
   connection the job talks to (paper port **4002**). *Not* TWS — Gateway uses
   far less memory and has no UI to babysit.
2. **IBC (IBController)** — auto-logs-in to Gateway and, critically, handles
   IBKR's **forced daily restart** (Gateway logs you out once every 24h). Without
   IBC the automation silently dies after a day.
3. **cron** — fires `auto_rebalance.py` every weekday at 10:00 ET. The job itself
   decides what's due (exposure check / quarterly rebalance / nothing).

```
  cron (weekday 10:00 ET)
        │  runs
        ▼
  auto_rebalance.py ──API──▶ IB Gateway ──▶ IBKR servers ──▶ paper account
        ▲                        ▲
        │ reads live prices      │ kept logged-in & auto-restarted by
        │ (yfinance)             │ IBC
   universe.csv, state           │
```

---

## 1. Hardware & OS

- **Raspberry Pi 4 or 5, 4GB+ RAM** (8GB comfortable). Gateway + Python need a
  few hundred MB; 2GB is tight, 4GB+ is safe.
- **64-bit Raspberry Pi OS (Bookworm) Lite** — headless, no desktop needed.
  Flash with Raspberry Pi Imager; enable SSH and set Wi-Fi/hostname in the
  Imager's advanced options.
- Wired Ethernet preferred over Wi-Fi for reliability.
- Give it a **static IP / DHCP reservation** on your router so you can always SSH in.

First boot:

```bash
ssh pi@<pi-ip>
sudo apt update && sudo apt full-upgrade -y
sudo raspi-config      # set locale, enable SSH if not already
```

### Timezone — the #1 silent scheduling bug

The cron entry below assumes the Pi's clock is on **US Eastern** so "10:00" means
10:00 ET during market hours. Set it explicitly:

```bash
sudo timedatectl set-timezone America/New_York
timedatectl            # verify; note DST is handled automatically
```

(If you'd rather keep the Pi on UTC, translate 10:00 ET yourself and remember it
shifts an hour across daylight-saving changes — easier to just set ET.)

---

## 2. Install Java + IB Gateway (the ARM caveat)

IBKR ships IB Gateway with a bundled **x86** Java runtime, which will **not** run
on the Pi's ARM chip. The fix is to install a system ARM Java and run Gateway's
installer with it (it detects an existing JVM and skips the bundled one), or run
the unpacked app under system Java.

```bash
# ARM Java
sudo apt install -y openjdk-17-jre xvfb x11vnc   # xvfb = headless virtual display

# Download the LATEST Gateway (stable) for Linux x64 — it's Java, arch-independent
# once you use system Java. IBC works most reliably with the "stable" channel.
cd ~
wget https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh
chmod +x ibgateway-stable-standalone-linux-x64.sh
# Run the installer with system Java so it doesn't try the bundled x86 JRE:
./ibgateway-stable-standalone-linux-x64.sh   # installs to ~/Jts or ~/ibgateway
```

If the installer refuses the bundled JRE on ARM, install to a folder and point
IBC at the app's `.jar` using `openjdk-17`. IBKR + IBC on ARM is well-documented;
search "IBC Raspberry Pi IB Gateway ARM" if a step fights you.

> **Simpler alternative if ARM Java fights you:** run this on a cheap always-on
> x86 mini-PC (or a small $5/mo cloud VM) instead of the Pi. Everything else in
> this guide is identical; you skip the ARM-Java dance entirely. The Pi is the
> cheapest/most-in-your-control option, an x86 box is the most turnkey.

Gateway needs a display even when headless — that's what `xvfb` is for (a virtual
framebuffer). IBC's start script wires this up for you.

---

## 3. Install & configure IBC (auto-login + daily restart)

IBC (IBController's maintained fork) logs into Gateway automatically and restarts
it after IBKR's daily logout.

```bash
cd ~
wget https://github.com/IbcAlpha/IBC/releases/latest/download/IBCLinux-3.x.x.zip
mkdir -p ~/ibc && unzip IBCLinux-*.zip -d ~/ibc
```

Edit `~/ibc/config.ini`:

```ini
IbLoginId=<your PAPER username>       # the paper login, NOT your live username
IbPassword=<your PAPER password>
TradingMode=paper
IbDir=/home/pi/Jts                    # where Gateway installed
FIX=no
# Let IBKR's nightly restart happen; IBC brings Gateway back up:
ClosedownAt=
AutoRestartTime=11:45 PM
# API:
OverrideTwsApiPort=4002
AcceptIncomingConnectionAction=accept
ReadOnlyApi=no
```

Store the password carefully — this file has your credentials; `chmod 600
~/ibc/config.ini` and keep the Pi off the public internet (see §7).

Start Gateway via IBC (headless, under xvfb). Test it manually first:

```bash
cd ~/ibc
xvfb-run ./gatewaystart.sh
# In another SSH session, confirm it's listening:
ss -ltnp | grep 4002
```

Once it stays up and auto-logs-in, run it as a **systemd service** so it launches
at boot and restarts on crash:

```ini
# /etc/systemd/system/ibgateway.service
[Unit]
Description=IB Gateway via IBC
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
ExecStart=/usr/bin/xvfb-run /home/pi/ibc/gatewaystart.sh
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ibgateway
sudo systemctl status ibgateway
```

**API settings inside Gateway** (IBC can set these, or set once via VNC):
Configure → Settings → API → Enable ActiveX and Socket Clients, **Socket port
4002**, add **127.0.0.1** to Trusted IPs, and enable "Allow connections from
localhost only." The job connects on `127.0.0.1:4002`, so nothing needs to be
exposed off-box.

---

## 4. Copy the project & set up Python

The runtime job (`auto_rebalance.py`) pulls **live prices via yfinance** at run
time, so it does **not** need the big backtest pickles (`prices_pit.pkl`, etc.).
Minimal runtime set:

```
auto_rebalance.py  data.py  sector_select.py  costs.py
paper_trader.py    ibkr.py  universe.csv
```

Simplest is just to copy the whole folder. From the Mac:

```bash
rsync -av --exclude '*.pkl' --exclude 'figures*' --exclude '__pycache__' \
  ~/Documents/autoPortfolio/  pi@<pi-ip>:~/autoPortfolio/
```

(Excluding `*.pkl` skips ~90MB of backtest data the live job doesn't use. Drop
that exclude if you also want to run backtests on the Pi.)

Python env on the Pi:

```bash
sudo apt install -y python3-venv python3-pip
cd ~/autoPortfolio
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install ibapi pandas numpy scipy scikit-learn yfinance
# (or: pip install -r requirements.txt)
```

Verified working versions on the Mac (targets, not hard pins): ibapi 9.81.1,
pandas 2.1.3, numpy 1.26.4, scipy 1.10.1, scikit-learn 1.3.2, yfinance 0.2.65.
On ARM these install from prebuilt wheels; if numpy/scipy try to compile, run
`sudo apt install -y libatlas-base-dev gfortran` first.

---

## 5. Point the job at Gateway (port 4002) and test

`auto_rebalance.py` already supports Gateway via `--gateway` (port 4002) instead
of TWS (7497). Test the ladder — **dry run first**, then live:

```bash
cd ~/autoPortfolio && . .venv/bin/activate

# 1) API connectivity only (read-only: account + positions):
python3 ibkr.py 4002

# 2) Full pipeline, computes + logs orders but TRANSMITS NOTHING:
python3 auto_rebalance.py --ibkr --gateway            # dry-run is the default

# 3) When you're satisfied, the live line (routes paper orders):
python3 auto_rebalance.py --ibkr --gateway --live
```

Check `decision_log.md` after each — every decision is written in plain English.

---

## 6. Schedule it with cron

Wrapper script `~/autoPortfolio/run_pi.sh`:

```bash
#!/bin/bash
cd /home/pi/autoPortfolio || exit 1
source .venv/bin/activate
echo "=== $(date) : auto_rebalance run ===" >> auto_run.log
python3 auto_rebalance.py --ibkr --gateway --live >> auto_run.log 2>&1
```

```bash
chmod +x ~/autoPortfolio/run_pi.sh
crontab -e
```

Add (weekdays 10:00, Pi clock already on ET from §1):

```cron
# min hour dom mon dow(1-5 = Mon-Fri)
0 10 * * 1-5  /home/pi/autoPortfolio/run_pi.sh
```

That's the whole scheduler — same weekday-10:00-ET cadence as the Mac launchd
job, minus the TCC problem. The job's own logic (`is_rebalance_day`,
`is_exposure_check_day`) decides quarterly-vs-weekly-vs-nothing; `auto_state.json`
persists the two clocks across reboots.

---

## 7. Security & reliability notes

- **Never expose the API port off the Pi.** Keep Gateway bound to `127.0.0.1`
  and trusted-IP 127.0.0.1 only. Don't port-forward 4002. The cron job runs on
  the same box, so it never needs the network.
- **Credentials:** `config.ini` holds your paper password — `chmod 600`, and
  don't commit it. (Paper only here, but still.)
- **Firewall:** `sudo apt install ufw && sudo ufw allow ssh && sudo ufw enable`.
- **Power:** use a good 5V/3A supply and a quality SD card (or better, boot from
  USB SSD — SD cards wear out under constant logging). Consider a small UPS.
- **Watchdog:** systemd `Restart=always` (§3) brings Gateway back after crashes;
  IBC brings it back after IBKR's nightly logout.
- **Log rotation:** `auto_run.log` / `decision_log.*` grow slowly (a few lines a
  week), but add a `logrotate` rule if you like tidy.

---

## 8. Go-live checklist

- [ ] Pi on 64-bit OS Lite, static IP, timezone = America/New_York
- [ ] `openjdk-17`, `xvfb` installed; IB Gateway installed under system Java
- [ ] IBC `config.ini` set to **paper** creds, port 4002, `chmod 600`
- [ ] `ibgateway.service` enabled; `ss -ltnp | grep 4002` shows it listening
- [ ] Gateway API: socket 4002, trusted IP 127.0.0.1, localhost-only
- [ ] Project synced; venv created; `python3 ibkr.py 4002` prints account + 30 positions
- [ ] Dry run reviewed in `decision_log.md`, then `--live` verified
- [ ] cron entry `0 10 * * 1-5` installed; `run_pi.sh` executable
- [ ] Reboot the Pi and confirm Gateway auto-starts and API answers

Once this is green, the Mac launchd agent should be **unloaded** so the two don't
both trade:

```bash
# on the Mac:
launchctl unload ~/Library/LaunchAgents/com.autoportfolio.rebalance.plist
```

---

## 9. Watching it from your phone

You can monitor (and manually trade) the same paper account from the **IBKR
Mobile** app — see the note in the main chat about the *one-session-per-username*
gotcha and the fix (request a second username for the account so the phone login
doesn't bump the Pi's Gateway session). The Pi runs the automation; the phone is
your read-only dashboard.
