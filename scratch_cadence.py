import numpy as np, pandas as pd
import final_strategy as fs
import exposure_models as em
from backtester import performance

d = fs.build()
raw = d["raw"].dropna()
rv = em.realized_vol_series(raw)
target = em.exposure_voltarget(rv, target=0.15)   # continuous daily target

def turn(e):
    return e.diff().abs().sum() / (len(e) / 252)

rows = {}
def rec(name, e):
    r, e = em.apply_exposure(raw, e)
    m = performance(r); mt = performance(r.loc["2024-12-31":])
    rows[name] = {"Sharpe": m["sharpe"], "CAGR": m["cagr"], "Vol": m["vol"],
                  "MaxDD": m["max_dd"], "TestSharpe": mt["sharpe"],
                  "ExpTurn": turn(e)}

rec("continuous daily (validated)", target)
rec("daily, band 10%", em.banded_exposure(target, band=0.10, check_every=1))
rec("weekly, band 10%", em.banded_exposure(target, band=0.10, check_every=5))
rec("weekly, band 5%", em.banded_exposure(target, band=0.05, check_every=5))
rec("weekly, band 15%", em.banded_exposure(target, band=0.15, check_every=5))
rec("monthly, band 10%", em.banded_exposure(target, band=0.10, check_every=21))
rec("quarterly only", em.banded_exposure(target, band=0.0, check_every=63))
# reference: no overlay
r0 = performance(raw)
rows["raw (no overlay)"] = {"Sharpe": r0["sharpe"], "CAGR": r0["cagr"],
                            "Vol": r0["vol"], "MaxDD": r0["max_dd"],
                            "TestSharpe": performance(raw.loc["2024-12-31":])["sharpe"],
                            "ExpTurn": 0.0}

tbl = pd.DataFrame(rows).T[["Sharpe","CAGR","Vol","MaxDD","TestSharpe","ExpTurn"]]
print("EXPOSURE-CADENCE BAKE-OFF (locked liquid-30 book, 15% vol-target)\n")
h = "{:28s} {:>7} {:>7} {:>7} {:>7} {:>9} {:>8}"
print(h.format("cadence","Sharpe","CAGR","Vol","MaxDD","TestShrp","ExpTurn"))
for name,row in tbl.iterrows():
    print("{:28s} {:>7.2f} {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>9.2f} {:>7.1f}x".format(
        name,row["Sharpe"],row["CAGR"]*100,row["Vol"]*100,row["MaxDD"]*100,
        row["TestSharpe"],row["ExpTurn"]))
