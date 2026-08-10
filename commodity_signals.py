"""
commodity_signals.py — does a news/fundamental/positioning signal PREDICT next
month's commodity returns? The evaluation harness, shared by every factor so a
"win" is signal, not accounting.

The discipline that has rejected most of what we've tried (equity sentiment, FX
sentiment, PEAD) applies here in full:

  * NO LOOKAHEAD — a signal known at month t is only ever paired with the return
    realized over (t, t+1]. `_align` enforces this (fwd = rets.shift(-1)).
  * CROSS-SECTIONAL IC — each month, rank-correlate the signal across commodities
    with next month's return across commodities; average over months.
  * A TILT BACKTEST — long the top-signal commodities, short the bottom, realize
    the forward return (the tradable version of the IC).
  * A NULL — within each month permute the signal across commodities (destroying
    the signal->return correspondence while preserving every value and the NaN
    structure), recompute the statistic hundreds of times. A real signal must sit
    well outside this null; a random signal sits inside it (that is the whole
    point — it's what caught equity/FX sentiment being noise).

Returns are the monthly commodity panel from commodity_data (or the real futures
book later); the harness is agnostic to where the signal came from.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PPY = 12


# --- alignment (the no-lookahead guarantee) --------------------------------
def _align(signal: pd.DataFrame, rets: pd.DataFrame):
    """Pair signal_t with the return realized over (t, t+1]. fwd_t = rets_{t+1}.
    Both returned on signal's index, columns intersected."""
    cols = [c for c in signal.columns if c in rets.columns]
    common = signal.index.intersection(rets.index)
    sig = signal.loc[common, cols]
    fwd = rets.shift(-1).loc[common, cols]      # NEXT month's return
    return sig, fwd


# --- information coefficient -----------------------------------------------
def information_coefficient(signal, rets, method="spearman") -> pd.Series:
    """Per-month cross-sectional IC (rank corr of signal vs next-month return)."""
    sig, fwd = _align(signal, rets)
    ics = {}
    for t in sig.index:
        pair = pd.concat([sig.loc[t], fwd.loc[t]], axis=1).dropna()
        if len(pair) >= 3:
            ics[t] = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)
    return pd.Series(ics, name="ic")


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    if len(ic) < 3:
        return {"mean_ic": np.nan, "ic_ir": np.nan, "ic_t": np.nan,
                "hit": np.nan, "n": len(ic)}
    mean, sd = ic.mean(), ic.std(ddof=1)
    return {"mean_ic": mean,
            "ic_ir": mean / sd if sd > 0 else np.nan,             # per-period ratio
            "ic_t": mean / sd * np.sqrt(len(ic)) if sd > 0 else np.nan,  # t-stat
            "hit": (ic > 0).mean(), "n": len(ic)}


# --- tilt backtest (the tradable version) ----------------------------------
def tilt_backtest(signal, rets, mode="ls", frac=0.34, cost_bps=0.0) -> pd.Series:
    """
    Cross-sectional tilt: each month rank commodities by the signal, long the top
    `frac`, short the bottom `frac` (mode='ls') or long-only the top (mode='long'),
    equal-weight, realize NEXT month's return. Optional turnover cost.
    """
    sig, fwd = _align(signal, rets)
    prev_w = pd.Series(0.0, dtype=float)
    rows = {}
    for t in sig.index:
        s = sig.loc[t].dropna()
        if len(s) < 4:
            continue
        n = max(1, int(round(len(s) * frac)))
        order = s.sort_values()
        shorts, longs = order.index[:n], order.index[-n:]
        w = pd.Series(0.0, index=s.index)
        if mode == "ls":
            w[longs] = 0.5 / n
            w[shorts] = -0.5 / n
        else:
            w[longs] = 1.0 / n
        f = fwd.loc[t].reindex(w.index).fillna(0.0)
        gross = float((w * f).sum())
        turn = float((w - prev_w.reindex(w.index).fillna(0.0)).abs().sum())
        rows[t] = gross - turn * cost_bps / 1e4
        prev_w = w
    return pd.Series(rows, name="tilt")


def performance(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 6:
        return {k: np.nan for k in ("sharpe", "vol", "hit", "mean_mo", "n")}
    mu, sd = r.mean(), r.std(ddof=1)
    return {"sharpe": mu / sd * np.sqrt(PPY) if sd > 0 else np.nan,
            "vol": sd * np.sqrt(PPY), "hit": (r > 0).mean(),
            "mean_mo": mu, "n": len(r)}


# --- time-series / pooled evaluation (for narrow, sector-specific signals) --
def pooled_ic(signal, rets, method="spearman") -> float:
    """
    Pooled/time-series IC: stack every (signal_{t,c}, fwd_ret_{t,c}) pair across
    commodities AND months into one rank correlation. This is the right test for
    NARROW signals (1–2 commodities, e.g. EIA oil/gas inventories, natgas weather)
    where a cross-section of ≥3 names doesn't exist — it asks "when this signal is
    high, does that market's own next-month return tend to be high?"
    """
    sig, fwd = _align(signal, rets)
    pairs = pd.concat([sig.stack(dropna=False), fwd.stack(dropna=False)],
                      axis=1).dropna()
    if len(pairs) < 12:
        return np.nan
    return pairs.iloc[:, 0].corr(pairs.iloc[:, 1], method=method)


def timeseries_tilt(signal, rets, cost_bps=0.0) -> pd.Series:
    """
    Directional time-series book: position_{t,c} = sign(signal_{t,c}); realize the
    NEXT month's return; equal-weight across the commodities that have a signal.
    The tradable form of pooled IC (works for a single commodity too).
    """
    sig, fwd = _align(signal, rets)
    pos = np.sign(sig)
    strat = (pos * fwd).mean(axis=1, skipna=True)
    return strat.dropna().rename("ts_tilt")


def _permute_within_commodity(sig: pd.DataFrame, rng) -> pd.DataFrame:
    """Shuffle each commodity's signal down its OWN time axis (breaks the temporal
    signal->return link, preserves each column's value distribution + NaNs). The
    null for pooled/time-series IC."""
    out = sig.copy()
    arr = out.values
    for j in range(arr.shape[1]):
        col = arr[:, j]
        mask = ~pd.isna(col)
        if mask.sum() > 1:
            vals = col[mask].copy()
            rng.shuffle(vals)
            col[mask] = vals
    return out


def null_test_ts(signal, rets, n_null=300, seed=0, method="spearman") -> dict:
    """Pooled-IC and time-series-tilt Sharpe vs the within-commodity permutation
    null (shuffle each column's time order)."""
    real_ic = pooled_ic(signal, rets, method)
    real_sh = performance(timeseries_tilt(signal, rets))["sharpe"]
    sig, _ = _align(signal, rets)
    rng = np.random.default_rng(seed)
    nic, nsh = [], []
    for _ in range(n_null):
        perm = _permute_within_commodity(sig, rng)
        nic.append(pooled_ic(perm, rets, method))
        nsh.append(performance(timeseries_tilt(perm, rets))["sharpe"])
    nic = np.array(nic, float); nic = nic[~np.isnan(nic)]
    nsh = np.array(nsh, float); nsh = nsh[~np.isnan(nsh)]

    def _z(real, dist):
        return (real - dist.mean()) / dist.std() if len(dist) and dist.std() > 0 else np.nan
    return {"pooled_ic": real_ic,
            "pooled_ic_z": _z(real_ic, nic) if len(nic) else np.nan,
            "pooled_pctile": float((nic < real_ic).mean()) if len(nic) else np.nan,
            "ts_sharpe": real_sh,
            "ts_sharpe_z": _z(real_sh, nsh) if len(nsh) else np.nan}


# --- the NULL (permute the signal within each month) -----------------------
def _permute_within_month(sig: pd.DataFrame, rng) -> pd.DataFrame:
    """Shuffle each month's signal values across commodities, preserving the NaN
    pattern and the exact value multiset — kills the signal<->return link only."""
    out = sig.copy()
    arr = out.values
    for i in range(arr.shape[0]):
        row = arr[i]
        mask = ~pd.isna(row)
        if mask.sum() > 1:
            vals = row[mask].copy()
            rng.shuffle(vals)
            row[mask] = vals
    return out


def null_test(signal, rets, n_null=300, seed=0, method="spearman",
              mode="ls", frac=0.34) -> dict:
    """
    Compare the REAL mean-IC and tilt-Sharpe against the within-month-permutation
    null distribution. Returns z-scores and percentiles; a genuine signal has
    z >> 0 and pctile near 1.0, a noise signal sits at z~0 / pctile~0.5.
    """
    real_ic = information_coefficient(signal, rets, method).mean()
    real_sh = performance(tilt_backtest(signal, rets, mode=mode, frac=frac))["sharpe"]
    sig, _ = _align(signal, rets)
    rng = np.random.default_rng(seed)
    null_ic, null_sh = [], []
    for _ in range(n_null):
        perm = _permute_within_month(sig, rng)
        null_ic.append(information_coefficient(perm, rets, method).mean())
        null_sh.append(performance(tilt_backtest(perm, rets, mode=mode,
                                                 frac=frac))["sharpe"])
    null_ic = np.array(null_ic, float)
    null_sh = np.array(null_sh, float)
    null_sh = null_sh[~np.isnan(null_sh)]

    def _z(real, dist):
        sd = dist.std()
        return (real - dist.mean()) / sd if sd > 0 else np.nan

    return {
        "real_mean_ic": real_ic, "null_ic_mean": null_ic.mean(),
        "null_ic_std": null_ic.std(), "ic_z": _z(real_ic, null_ic),
        "ic_pctile": float((null_ic < real_ic).mean()),
        "real_tilt_sharpe": real_sh,
        "null_sharpe_mean": null_sh.mean() if len(null_sh) else np.nan,
        "tilt_z": _z(real_sh, null_sh) if len(null_sh) else np.nan,
        "tilt_pctile": float((null_sh < real_sh).mean()) if len(null_sh) else np.nan,
        "n_null": n_null,
    }


# --- one-call evaluation of a signal ---------------------------------------
def evaluate_signal(signal, rets, name="signal", n_null=300, seed=0,
                    method="spearman", frac=0.34) -> dict:
    """
    Evaluate a signal both ways and let the verdict use whichever fits:
      * TIME-SERIES / pooled (always) — right for narrow, sector-specific signals.
      * CROSS-SECTIONAL (only when ≥3 commodities have data each month) — the
        relative-ranking question.
    `passes` = a positive IC clearing z>2 under EITHER test.
    """
    out = {"name": name, "cols": int(signal.shape[1])}

    # time-series / pooled — universal (works for 1..N commodities)
    ts = null_test_ts(signal, rets, n_null=n_null, seed=seed, method=method)
    out.update({"pooled_ic": ts["pooled_ic"], "pooled_ic_z": ts["pooled_ic_z"],
                "pooled_pctile": ts["pooled_pctile"],
                "ts_sharpe": ts["ts_sharpe"], "ts_sharpe_z": ts["ts_sharpe_z"]})

    # cross-sectional — only if a real cross-section exists
    has_xs = (signal.notna().sum(axis=1) >= 3).sum() >= 12
    if has_xs:
        ic = information_coefficient(signal, rets, method)
        cs = ic_summary(ic)
        ncs = null_test(signal, rets, n_null=n_null, seed=seed, method=method,
                        mode="ls", frac=frac)
        out.update({"xs_mean_ic": cs["mean_ic"], "xs_ic_t": cs["ic_t"],
                    "xs_ic_z": ncs["ic_z"], "xs_pctile": ncs["ic_pctile"],
                    "xs_tilt_sharpe": ncs["real_tilt_sharpe"], "n_xs": cs["n"]})

    ts_pass = (out.get("pooled_ic_z") or -9) > 2 and (out.get("pooled_ic") or 0) > 0
    xs_pass = (out.get("xs_ic_z") or -9) > 2 and (out.get("xs_mean_ic") or 0) > 0
    out["passes"] = bool(ts_pass or xs_pass)
    return out


def evaluate_all(signal_panels: dict, rets, **kw) -> pd.DataFrame:
    """Evaluate several named signal panels; one row each, most-significant first
    (by the stronger of the pooled / cross-sectional z-score)."""
    rows = [evaluate_signal(p, rets, name=nm, **kw)
            for nm, p in signal_panels.items() if p is not None and len(p)]
    df = pd.DataFrame(rows).set_index("name")
    df["best_z"] = df[["pooled_ic_z", "xs_ic_z"]].max(axis=1) \
        if "xs_ic_z" in df else df["pooled_ic_z"]
    return df.sort_values("best_z", ascending=False)
