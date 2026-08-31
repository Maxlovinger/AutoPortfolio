"""
portfolio_snapshot.py — read the paper account through the Pi's existing Gateway
session and emit a phone-friendly HTML+text snapshot: NAV, 3-sleeve breakdown, P&L,
and positions. NEVER opens a second IBKR login (reads the session already running),
so it can't disturb the automation.

Modes:
  (default)  print HTML to stdout (for capture / testing)
  --send     email via SMTP using env: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS,
             SNAPSHOT_TO  (set in ~/autoPortfolio/.env)
"""
from __future__ import annotations
import os, sys, ssl, time, threading, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper

BONDS = {"IEF","TLT","SHY","AGG","BND","GOVT","IEI","TLH"}
COUNTRY = {
    "USD":"United States","EUR":"Euro area","JPY":"Japan","GBP":"United Kingdom",
    "CHF":"Switzerland","AUD":"Australia","NZD":"New Zealand","CAD":"Canada",
    "SEK":"Sweden","NOK":"Norway","DKK":"Denmark","MXN":"Mexico","ZAR":"South Africa",
    "HUF":"Hungary","PLN":"Poland","CZK":"Czechia","TRY":"Turkey","BRL":"Brazil",
    "INR":"India","KRW":"South Korea","CLP":"Chile","ILS":"Israel","CNH":"China",
    "SGD":"Singapore","HKD":"Hong Kong","THB":"Thailand","IDR":"Indonesia",
    "PHP":"Philippines","COP":"Colombia","PEN":"Peru",
}
INFO = {2104,2106,2107,2108,2119,2158,2100,2150,2168,2169}
PORT = 4002


class Snap(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.v = {}; self.stk = {}; self.fx = []; self.acctdone=False; self.posdone=False
        self.acct=None; self._acctev=threading.Event()
        # daily / unrealized / realized from reqPnL (authoritative, real-time)
        self.pnl_daily=None; self.pnl_unreal=None; self.pnl_real=None
    def managedAccounts(self, accountsList):
        self.acct = accountsList.split(",")[0].strip()
        self._acctev.set()
    def error(self, reqId, code="", msg="", *a):
        if code not in INFO: print(f"IB {code}: {msg}", file=sys.stderr)
    def updateAccountValue(self, k, val, cur, acct):
        # IB streams unrealized/realized P&L under "$LEDGER-UnrealizedPnL" /
        # "$LEDGER-RealizedPnL" — NOT plain "UnrealizedPnL" (which never arrives,
        # which is why the report showed $0). Normalize the ledger keys here so
        # they're a reliable fallback if reqPnL doesn't respond.
        key = {"$LEDGER-UnrealizedPnL": "UnrealizedPnL",
               "$LEDGER-RealizedPnL": "RealizedPnL"}.get(k, k)
        if cur in ("USD","BASE") and key in ("NetLiquidation","TotalCashValue",
                                             "UnrealizedPnL","RealizedPnL","AvailableFunds"):
            try: self.v[key] = float(val)
            except: pass
    def updatePortfolio(self, c, pos, mp, mv, avg, upnl, rpnl, acct):
        if pos == 0: return
        if c.secType == "STK":
            self.stk[c.symbol] = {"pos":pos,"mv":float(mv),"upnl":float(upnl)}
    def accountDownloadEnd(self, a): self.acctdone=True
    def pnl(self, reqId, dailyPnL, unrealizedPnL, realizedPnL):
        self.pnl_daily=dailyPnL; self.pnl_unreal=unrealizedPnL; self.pnl_real=realizedPnL
    def position(self, acct, c, pos, avg):
        if c.secType=="CASH" and pos!=0:
            self.fx.append((c.symbol, c.currency, float(pos)))
    def positionEnd(self): self.posdone=True


def gather():
    app = Snap(); app.connect("127.0.0.1", PORT, clientId=19)
    threading.Thread(target=app.run, daemon=True).start()
    app._acctev.wait(6)                       # account code arrives on connect
    app.reqAccountUpdates(True, ""); app.reqPositions()
    if app.acct:
        app.reqPnL(7010, app.acct, "")        # daily + unrealized + realized P&L
    t=time.time()
    while (not app.acctdone or not app.posdone or app.pnl_daily is None) \
            and time.time()-t<20: time.sleep(0.3)
    time.sleep(1)
    try: app.cancelPnL(7010)
    except Exception: pass
    app.disconnect()
    return app


def resolve_pnl(app):
    """Best-available (daily, unrealized) P&L. Prefer reqPnL (real-time, excludes
    deposits); fall back to the account ledger, then per-position marks."""
    unreal = app.pnl_unreal
    if unreal is None:
        unreal = app.v.get("UnrealizedPnL")
    if unreal is None:
        unreal = sum(d["upnl"] for d in app.stk.values()) or None
    daily = app.pnl_daily                     # None if reqPnL didn't respond
    return daily, unreal


def build(app):
    nav = app.v.get("NetLiquidation",0.0); cash = app.v.get("TotalCashValue",0.0)
    daily, upnl = resolve_pnl(app)
    upnl = upnl if upnl is not None else 0.0
    eq = {s:d for s,d in app.stk.items() if s not in BONDS}
    bd = {s:d for s,d in app.stk.items() if s in BONDS}
    eq_v = sum(d["mv"] for d in eq.values()); bd_v = sum(d["mv"] for d in bd.values())
    # FX carry legs -> readable long/short per ccy
    fx_legs = []
    for sym,cur,pos in app.fx:
        if sym=="USD": ccy, side = cur, ("SHORT" if pos>0 else "LONG"); notional=abs(pos)
        else: ccy, side = sym, ("LONG" if pos>0 else "SHORT"); notional=abs(pos)
        fx_legs.append((ccy, side, notional))
    fx_gross = sum(n for _,_,n in fx_legs)
    pct = lambda x: (x/nav*100) if nav else 0
    ts = pd.Timestamp.now(tz="America/New_York").strftime("%a %b %d, %Y %-I:%M %p ET")

    def row(label, val, p, extra=""):
        return (f"<tr><td style='padding:6px 10px'>{label}</td>"
                f"<td style='padding:6px 10px;text-align:right'>${val:,.0f}</td>"
                f"<td style='padding:6px 10px;text-align:right'>{p:.1f}%</td>"
                f"<td style='padding:6px 10px;color:#666'>{extra}</td></tr>")

    pnl_color = "#137333" if upnl>=0 else "#a50e0e"
    day_color = "#137333" if (daily or 0)>=0 else "#a50e0e"
    day_html = (f'<b style="color:{day_color}">${daily:+,.0f}</b>'
                f' <span style="color:#888">({daily/nav*100:+.2f}%)</span>'
                if daily is not None and nav else "<span style='color:#888'>n/a</span>")
    html = f"""<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:520px;margin:auto">
<h2 style="margin:0 0 2px">Portfolio — DUP856421 (paper)</h2>
<div style="color:#666;font-size:13px;margin-bottom:14px">{ts}</div>
<div style="font-size:15px;margin-bottom:4px">Net Liquidation: <b>${nav:,.0f}</b></div>
<div style="font-size:15px;margin-bottom:4px">Today's P&L: {day_html}</div>
<div style="font-size:15px;margin-bottom:14px">Unrealized P&L:
  <b style="color:{pnl_color}">${upnl:,.0f}</b></div>
<table style="border-collapse:collapse;width:100%;font-size:14px;border:1px solid #eee">
<tr style="background:#f5f6f8;font-weight:600">
  <td style='padding:6px 10px'>Sleeve</td><td style='padding:6px 10px;text-align:right'>Value</td>
  <td style='padding:6px 10px;text-align:right'>% NAV</td><td style='padding:6px 10px'>Target</td></tr>
{row("Equity", eq_v, pct(eq_v), f"{len(eq)} names · tgt 63.9%×exp")}
{row("Bonds (IEF)", bd_v, pct(bd_v), "tgt 10%")}
{row("FX carry (gross)", fx_gross, pct(fx_gross), f"{len(fx_legs)} legs · $-neutral · tgt 26.1%")}
{row("Cash", cash, pct(cash), "de-risked slice")}
</table>
<h3 style="margin:16px 0 6px;font-size:15px">FX carry legs</h3>
<div style="font-size:14px;line-height:1.9">{' · '.join(f'<b>{s}</b> {c} <span style="color:#888">({COUNTRY.get(c,"?")})</span>' for c,s,_ in sorted(fx_legs, key=lambda x:x[1])) or 'none'}</div>
<h3 style="margin:16px 0 6px;font-size:15px">Equity + bonds ({len(eq)+len(bd)} positions)</h3>
<div style="font-size:13px;color:#333;line-height:1.7">
{' · '.join(f"{s} ${d['mv']:,.0f}" for s,d in sorted({**eq,**bd}.items(), key=lambda kv:-kv[1]['mv']))}
</div>
<div style="color:#999;font-size:11px;margin-top:16px">Read-only snapshot via the Pi's Gateway session — no second login, automation undisturbed.</div>
</div>"""

    day_txt = f"${daily:+,.0f} ({daily/nav*100:+.2f}%)" if daily is not None and nav else "n/a"
    text = (f"Portfolio DUP856421 (paper) — {ts}\n"
            f"NAV ${nav:,.0f} | Today's P&L {day_txt} | Unrealized P&L ${upnl:,.0f}\n"
            f"Equity ${eq_v:,.0f} ({pct(eq_v):.1f}%) | Bonds ${bd_v:,.0f} ({pct(bd_v):.1f}%) | "
            f"FX gross ${fx_gross:,.0f} ({pct(fx_gross):.1f}%) | Cash ${cash:,.0f} ({pct(cash):.1f}%)\n"
            f"FX: {', '.join(f'{s} {c} ({COUNTRY.get(c,chr(63))})' for c,s,_ in sorted(fx_legs,key=lambda x:x[1]))}\n")
    day_subj = f"Day ${daily:+,.0f}" if daily is not None else "Day n/a"
    subj = (f"Portfolio ${nav:,.0f} | {day_subj} | Unrl ${upnl:+,.0f} | "
            f"{pd.Timestamp.now(tz='America/New_York').strftime('%b %d')}")
    return subj, html, text


def send_email(subj, html, text):
    host=os.getenv("SMTP_HOST"); port=int(os.getenv("SMTP_PORT","465"))
    user=os.getenv("SMTP_USER"); pw=os.getenv("SMTP_PASS"); to=os.getenv("SNAPSHOT_TO", user)
    if not (host and user and pw):
        print("SMTP env not set (SMTP_HOST/SMTP_USER/SMTP_PASS) — cannot send.", file=sys.stderr)
        return False
    msg = MIMEMultipart("alternative"); msg["Subject"]=subj; msg["From"]=user; msg["To"]=to
    msg.attach(MIMEText(text,"plain")); msg.attach(MIMEText(html,"html"))
    with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context()) as s:
        s.login(user, pw); s.sendmail(user, [to], msg.as_string())
    print(f"snapshot emailed to {to}")
    return True


def main():
    # load .env for SMTP creds
    envp = Path(__file__).resolve().parent / ".env"
    if envp.exists():
        for line in envp.read_text().splitlines():
            line=line.strip()
            if line and not line.startswith("#") and "=" in line:
                k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    app = gather()
    subj, html, text = build(app)
    if "--send" in sys.argv:
        send_email(subj, html, text)
    else:
        print("SUBJECT:", subj); print(html)


if __name__ == "__main__":
    main()
