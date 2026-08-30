"""Exit 0 if the Gateway is logged in (NAV readable), else exit 1 (caller restarts)."""
import sys, threading, time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
class H(EWrapper, EClient):
    def __init__(s): EClient.__init__(s, s); s.nlv=None; s.done=False
    def updateAccountValue(s,k,v,cur,ac):
        if k=="NetLiquidation" and cur in ("USD","BASE"):
            try: s.nlv=float(v)
            except: pass
    def accountDownloadEnd(s,a): s.done=True
    def error(s,*a): pass
try:
    h=H(); h.connect("127.0.0.1",4002,88)
    threading.Thread(target=h.run,daemon=True).start(); time.sleep(2)
    h.reqAccountUpdates(True,"")
    t=time.time()
    while not h.done and time.time()-t<15: time.sleep(0.3)
    h.disconnect()
    ok = h.nlv is not None and h.nlv > 0
    print("healthy, NAV", h.nlv) if ok else print("UNHEALTHY nav=", h.nlv)
    sys.exit(0 if ok else 1)
except Exception as e:
    print("UNHEALTHY exc", e); sys.exit(1)
