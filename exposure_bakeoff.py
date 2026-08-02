import numpy as np, pandas as pd
import final_strategy as fs
import exposure_models as em
from backtester import performance

d = fs.build()
raw = d["raw"].dropna()
bench = d["bench"].reindex(raw.index).fillna(0.0)
TRAIN_END = "2022-12-31"

# --- Question A: MLE lambda on training window ---
lam = em.fit_ewma_lambda(raw.loc[:TRAIN_END])
print(f"[A] MLE-fit EWMA lambda (train only): {lam:.4f}  (RiskMetrics default 0.94)\n")

def turn(e):
    return e.diff().abs().sum() / (len(e) / 252)

results = {}

def record(name, r, e=None):
    full = performance(r)
    tst = performance(r.loc["2024-12-31":])
    results[name] = {"Sharpe": full["sharpe"], "CAGR": full["cagr"],
                     "Vol": full["vol"], "MaxDD": full["max_dd"],
                     "TestSharpe": tst["sharpe"],
                     "ExpTurn": turn(e) if e is not None else np.nan}

# baselines
record("raw (100% invested)", raw)

# vol-target family
rv = em.realized_vol_series(raw)
r, e = em.apply_exposure(raw, em.exposure_voltarget(rv)); record("voltgt realized", r, e)
ew94 = em.ewma_vol_series(raw, 0.94)
r, e = em.apply_exposure(raw, em.exposure_voltarget(ew94)); record("voltgt ewma-0.94", r, e)
ewml = em.ewma_vol_series(raw, lam)
r, e = em.apply_exposure(raw, em.exposure_voltarget(ewml)); record("voltgt ewma-MLE", r, e)

# trend
r, e = em.apply_exposure(raw, em.exposure_trend(raw, window=200, floor=0.0)); record("trend-200 (cash)", r, e)
r, e = em.apply_exposure(raw, em.exposure_trend(raw, window=200, floor=0.5)); record("trend-200 (floor.5)", r, e)

# regime
r, e = em.apply_exposure(raw, em.exposure_regime(raw, TRAIN_END, floor=0.0)); record("regime-HMM", r, e)

# CPPI
r, e = em.cppi_returns(raw, m=3.0, floor_frac=0.80); record("cppi m3 floor80", r, e)
r, e = em.cppi_returns(raw, m=4.0, floor_frac=0.85); record("cppi m4 floor85", r, e)

# combos: stack a signal with vol-target (min of the two exposures)
vt = em.exposure_voltarget(rv)
tr = em.exposure_trend(raw, 200, 0.0)
rg = em.exposure_regime(raw, TRAIN_END, 0.0)
r, e = em.apply_exposure(raw, np.minimum(vt, tr)); record("voltgt x trend", r, e)
r, e = em.apply_exposure(raw, np.minimum(vt, rg)); record("voltgt x regime", r, e)

tbl = pd.DataFrame(results).T
order = ["Sharpe", "CAGR", "Vol", "MaxDD", "TestSharpe", "ExpTurn"]
tbl = tbl[order]
print("EXPOSURE-MODEL BAKE-OFF (locked liquid-30 book; train<=2022, test 2025+)\n")
hdr = "{:22s} {:>7} {:>7} {:>7} {:>7} {:>9} {:>8}"
print(hdr.format("model", "Sharpe", "CAGR", "Vol", "MaxDD", "TestShrp", "ExpTurn"))
for name, row in tbl.iterrows():
    et = f"{row['ExpTurn']:6.1f}x" if np.isfinite(row["ExpTurn"]) else "     -"
    print("{:22s} {:>7.2f} {:>6.1f}% {:>6.1f}% {:>6.1f}% {:>9.2f} {:>8}".format(
        name, row["Sharpe"], row["CAGR"]*100, row["Vol"]*100, row["MaxDD"]*100,
        row["TestSharpe"], et))
