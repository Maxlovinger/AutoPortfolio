% Currency Trading Strategy — Plain-English Summary
% autoPortfolio (FX sleeve)
% 2026-08-08

---

## In one sentence

We try to earn the **"carry" premium** in currencies: hold the currencies that
pay high interest, fund them by shorting the currencies that pay low interest,
and keep the book market-neutral — then manage the risk honestly.

---

## What "carry" means (the whole idea)

Every currency pays an interest rate (its central bank's rate). If you hold a
high-rate currency and short a low-rate one, you **collect the difference** just
for holding the position — like earning the gap between two savings accounts.

In theory the high-rate currency should weaken enough to cancel that gain. In
practice it usually **doesn't** (a well-documented market quirk). That gap is the
premium we harvest. It is real but modest, and it comes with occasional sharp
losses ("carry crashes") when nervous investors flee to safe currencies.

---

## How the strategy works

1. **Universe:** the 10 major currencies (USD, EUR, JPY, GBP, CHF, AUD, NZD,
   CAD, SEK, NOK).
2. **Rank** them each month by their interest rate vs. the US dollar.
3. **Go long** the top few (highest rates), **short** the bottom few (lowest).
   Equal weight, dollar-neutral (no bet on the dollar overall).
4. **Rebalance monthly.** Costs are subtracted realistically.

That's it. Simple on purpose — we only add complexity if it *proves* it helps.

---

## What we tested, and what we found

We tested each idea head-to-head against plain carry. The rule: **it has to beat
plain carry after costs, or it's out.** Sharpe ratio = return per unit of risk
(higher is better).

| What we tried | Result | Kept? |
|---|---|---|
| **Plain carry, monthly** | Sharpe **0.31** — the benchmark | **Yes** |
| Rebalance weekly instead | Sharpe 0.27, worse | No |
| Add momentum + value signals | Sharpe 0.23, more trading, worse tail | No |
| Vol-target "crash" overlay | No better; made the crash risk *worse* | No |

**Three add-ons were tested. All three lost to plain carry.** Every time, the
extra complexity added trading costs and noise faster than it added edge.

**Why the crash overlay failed (important):** it cuts risk when markets get
volatile — but currency crashes are *sudden jumps* (e.g. the 2019 yen flash
crash) with no warning. The overlay was fully invested going *into* the crash
and only pulled back *after*, making things worse. An outside research paper we
reviewed predicted exactly this.

---

## Where the data comes from (all free)

- **Exchange rates:** Yahoo Finance (yfinance).
- **Interest rates:** FRED (US Federal Reserve database, free key).
- **Carry signal** = foreign rate − US rate. No paid data needed.

Caveat: the free interest-rate data lags 1–7 months — fine for research, but a
live version would need a fresher rate feed.

---

## News sentiment (new, optional add-on)

We can also read the *tone of the news* about each economy and use it as an extra
signal:

- **Scoring:** FinBERT, an AI model trained on financial text (rates
  "beats guidance" as positive, "fraud probe" as negative).
- **History:** GDELT — free, timestamped news tone, so we can *backtest* whether
  it actually helps before trusting it.
- Same discipline applies: it only stays if it beats plain carry after costs.

The same sentiment tooling also upgrades the **equity** portfolio's news signal.

---

## The honest bottom line

- **Plain monthly carry is the strategy.** It has a real, modest edge
  (Sharpe ≈ 0.3), in line with what professional research reports for major
  currencies.
- Everything fancier we tried made it **worse**, not better — so we didn't add
  it. (This matches the equity side, where simple has repeatedly beaten complex.)
- Carry's real weakness is the **crash tail**, not weak signal. The genuine next
  step is a signal that can *anticipate* those jumps (safe-haven flows, rate
  volatility, or the news-sentiment work above) — not more of the same.

**Status:** built and fully tested in code; validated on 2010–2026 data. Not yet
trading live.
