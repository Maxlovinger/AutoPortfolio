# Currency Model — Proposal

*Status: proposal / not yet built. Kept intentionally separate from the equity system (screener + Markowitz). Shares infrastructure patterns only: cross-sectional ranking, risk-scaled sizing, walk-forward validation with realistic costs.*

*Date: 2026-08-08*

---

## 1. Thesis

FX returns are dominated by **relative interest rates**, not price momentum in the equity sense. The tradable premium comes from a documented, persistent anomaly:

- **Uncovered Interest Parity (UIP) fails.** Theory says a high-yield currency should depreciate enough to cancel its yield advantage. Empirically it usually doesn't (the "forward premium puzzle"). That failure is *why* the carry trade has paid over decades.
- **Covered Interest Parity (CIP) holds.** Forward FX ≈ spot × (rate differential). So the forward already embeds the rate gap.

**Consequence — the core design constraint:** predicting the *level* of a policy rate is nearly worthless, because OIS curves, rate futures, and forward FX already price expected policy. The only tradable ML target is the **surprise / residual vs. market-implied expectations**. Everything below is built around that.

We do **not** build "predict interest rates → trade currency." We build a **cross-sectional currency ranker** whose features include the carry base plus an ML-estimated policy-surprise signal, sized down by a crash-risk overlay.

## 2. Universe

Start with liquid majors + a controlled EM sleeve:

- **G10:** USD, EUR, JPY, GBP, CHF, AUD, NZD, CAD, SEK, NOK
- **EM (optional, later):** MXN, ZAR, BRL, INR, PLN — where carry is juiciest but spreads/jump risk are worst. Gate behind separate cost + liquidity checks.

Trade as **excess return of currency i vs. USD** (or vs. an equal-weight basket for a dollar-neutral book).

## 3. Prediction target

For each currency i over horizon h (start monthly):

    y_i = spot return_i  +  carry earned_i   (i.e. total FX excess return)

Not the rate. Not the spot alone. The **carry-inclusive excess return**, which is what a funded position actually earns.

## 4. Features

**A. Carry base (the anchor, no ML needed)**
- Short-rate differential vs. USD (from OIS / 3M rates)
- Forward points (market-implied, sanity-checks the above)

**B. ML-estimated policy surprise (the edge)**
- Model each central bank's **reaction function**: map macro inputs → expected policy move, then take the **residual vs. market-implied path** (OIS-implied hikes/cuts, rate futures).
- Macro inputs: CPI surprise vs. consensus, core inflation trend, labor prints (unemployment, wage growth), PMIs, growth nowcasts.
- Nonlinear / regime-dependent by design (behavior differs near zero-bound, during inflation spikes) → gradient-boosted trees or a regime-switching layer, not plain linear.

**C. Classic FX style factors (documented premia)**
- **Value:** deviation from PPP fair value (mean-reversion signal)
- **Momentum:** 3–12M trailing spot/excess return
- **Dollar factor:** broad USD trend / risk-on-off proxy

**D. Risk / regime inputs (for the overlay, section 6)**
- FX-implied vol (majors), VIX, rate-vol (MOVE-style), realized-vol regime
- Carry-crash indicators: crowding, skew of carry basket returns

## 5. Model

- **Cross-sectional ranker**, not per-pair timers. Each period, rank the universe by predicted excess return; go long the top sleeve, short the bottom, dollar-neutral (or vs. USD).
- **Base learner:** gradient-boosted trees (handles nonlinear reaction functions + factor interactions) or a regularized linear combo of the style factors as a robust baseline to beat.
- **Blend:** carry + value + momentum + policy-surprise, weights learned but regularized hard (few effective parameters — FX has short history and violent regime shifts).

## 6. Risk overlay (the most important piece)

Carry's defining feature is **negative skew**: steady gains, then violent unwinds (Aug 2007, 2008, Jan 2015 CHF). Return prediction matters less than *not being fully exposed into a crash*.

- **Vol-target the book** to a fixed annualized risk.
- **Crash-risk scaler:** cut gross exposure when FX-implied vol / risk-off indicators / carry-crowding flash. This overlay is expected to add more risk-adjusted value than the return signal itself.
- Position limits per currency; tighter caps on EM.

## 7. Costs — non-negotiable, baked in from day one

- **Majors:** tight spreads, deep liquidity → the friendly case.
- **EM:** wide spreads, capital controls, gap/devaluation risk, gappy data — this is where paper edges die. Model spread + slippage per currency explicitly; do not let EM in until it survives realistic costs.
- Roll/funding costs for holding forwards.

## 8. Validation

Same discipline as the equity backtester — this is the gate that decides whether any of it is real:

- **Walk-forward** with **purged** train/test splits and an **embargo** period (no lookahead across the rate-decision boundary).
- **Deflated Sharpe ratio** — penalize for the number of configs tried.
- Report **skew, max drawdown, and worst-month** prominently, not just Sharpe — a good carry Sharpe hides tail risk.
- Benchmarks to beat: (a) naive equal-weight carry, (b) long-USD cash. If we can't beat plain carry after costs, the ML layer isn't earning its keep.

## 9. Data sources needed

- **Rates / policy expectations:** OIS curves per country, rate futures (Fed funds, SONIA, Euribor/€STR, etc.), 3M rates.
- **FX:** spot + forward points for the universe (majors free-ish; EM via a paid feed).
- **Macro:** CPI/labor/PMI actuals *and consensus* (for surprises) — consensus data is the expensive/hard part.

## 10. Build order

1. Assemble rates + spot/forward data for G10; compute carry base.
2. Backtest **plain carry** with realistic costs → establish the benchmark bar.
3. Add value + momentum factors; cross-sectional ranker.
4. Add the **risk/crash overlay**; check it improves skew & drawdown.
5. Only then add the **ML policy-surprise** feature (needs consensus macro data); measure incremental out-of-sample edge over steps 2–4.
6. Consider EM sleeve behind its own cost gate.

## 11. Honest expectation

- Plain carry: real long-run premium, low Sharpe (~0.4–0.6), ugly tails, crowded.
- ML on rate *levels*: **no edge** (already priced).
- ML on rate *surprises* + crash-risk timing: a **genuine, defensible** direction — close to what real systematic-macro desks do.
- The realistic win is a modest, better-behaved carry — not a high-Sharpe money printer. Success = beating naive carry *after costs* with lower drawdown.

## 12. Explicitly out of scope for now

- Kept separate from the equity portfolio; no shared capital allocation yet.
- No intraday / high-frequency FX — this is a monthly (or weekly) rebalanced systematic-macro sleeve.
- No single-pair directional betting.
