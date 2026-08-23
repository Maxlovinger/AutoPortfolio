% The Strategy, In Depth — A Plain-English Guide
% autoPortfolio: the 3-sleeve book (Equity + Currency + Bonds)
% 2026-08-09

# The big picture

You run **one portfolio made of three independent "sleeves."** Each does a
different job, and they're deliberately kept separate so that when one struggles,
the others don't have to.

| Sleeve | Weight | What it is | Its job |
|---|---|---|---|
| **Equity** | **63.9%** | 30 large US stocks | the **return engine** |
| **Currency (carry)** | **26.1%** | long high-rate / short low-rate currencies | **diversifier** |
| **Bonds** | **10.0%** | US Treasuries (IEF) | **shock absorber** |

Those three weights are the whole allocation. The rest of this guide explains,
in order: **what each sleeve holds**, then the two things people find most
confusing — **how exposure changes** and **when everything rebalances** — and
finally a **full worked month** so you can see it all move together.

\newpage

# Sleeve 1 — Equity (the engine, 63.9%)

## What it holds

The **30 most heavily-traded large US stocks**, each held in **equal size**
(about 1/30 of the sleeve), with **no more than 5 from any one industry**.

Right now that's names like Apple, Microsoft-scale companies, big banks,
healthcare, energy, industrials — 30 of them, spread across 8 sectors.

## Why these rules (each was tested, simple won)

- **Most-traded ("liquid") names** → cheap to buy and sell, realistic to run.
- **Equal weight (1/30 each)** → we tested "smarter" weighting schemes; plain
  equal weight beat all of them. This is a famous, hard-to-beat result.
- **Max 5 per sector** → stops the book quietly becoming an all-tech bet. Testing
  confirmed the edge is *being broad and liquid*, not a sector bet.

## The safety feature: it dials risk up and down

This is the **exposure** part (explained in full in its own section below). In
short: when markets get stormy, the equity sleeve **automatically moves part of
itself to cash** to keep risk steady; when calm, it's fully invested. It only
ever *reduces* risk — it never borrows to add more.

\newpage

# Sleeve 2 — Currency carry (the diversifier, 26.1%)

## How trading a currency works (the one-minute version)

You never own a currency by itself — you always hold one **against** another.
Every currency also pays an **interest rate** set by its central bank. That
interest rate is the whole game here.

## What it does

> Hold ("go **long**") the currencies that pay **high** interest, and borrow /
> sell short ("go **short**") the currencies that pay **low** interest. Pocket the
> difference in interest.

That interest gap is called **carry**. It's like borrowing where money is cheap
and depositing where it pays well, and keeping the spread.

- Doing both at once makes the sleeve **"dollar-neutral"** — equal money long and
  short — so you're **not** betting on the dollar overall, only on high-rate
  currencies beating low-rate ones.
- Right now the book is roughly: **long** Mexican peso, Hungarian forint, South
  African rand (high rates); **short** Swiss franc, Canadian dollar, Swedish krona
  (low rates).

## Why it diversifies

Its ups and downs have **almost nothing to do with the stock market** — and
crucially, in the stock market's *worst* months it does **not** crash alongside
stocks. That "independence when it matters" is exactly why it earns a place next
to the equity engine.

Its one real danger is a **"carry crash"**: high-rate currencies (especially
emerging ones) can occasionally drop sharply all at once. So it earns steady
small gains, then occasionally takes a quick hit — which is why we **size it
carefully** and keep it a satellite, not the main event.

\newpage

# Sleeve 3 — Bonds (the shock absorber, 10%)

## What it holds

A single, simple holding: **IEF**, an ETF of **US Treasury notes** (government
bonds, 7–10 year maturity). US government debt — the safest issuer there is.

## Why it's here

In a normal recession-type scare, investors flee to Treasuries and the central
bank cuts interest rates — so **bonds tend to rise exactly when stocks fall.**
That makes them a **shock absorber**: a small, steady holding that softens the
ride and pays interest (~4–5% a year right now) while it waits.

## The honest caveat

Bonds are a hedge against **growth** shocks, **not inflation** shocks. In 2022,
inflation spiked, the Fed hiked, and bonds fell *with* stocks. So we keep the
bond sleeve **small (10%)** and use **intermediate** maturity (not long-dated,
which gets hurt worse in an inflation shock). At 10% it makes the ride a little
smoother without dragging on returns much.

\newpage

# How exposure changes (the part everyone asks about)

"Exposure" = **how much of the equity sleeve is actually invested** versus parked
in cash. It's a dial from 0% to 100%, and it moves on its own.

## The rule, in words

> Measure how jumpy the equity book has been lately. If it's calmer than our 15%
> risk target, stay fully invested. If it's jumpier, invest **less** and hold the
> rest in cash — just enough to keep the risk near 15% a year.

The exact dial:

    exposure  =  the smaller of:  100%   or   15% ÷ (recent book volatility)

- **Calm markets** (volatility below 15%) → the formula wants more than 100%, but
  we cap at 100%. **Fully invested.**
- **Turbulent markets** (volatility above 15%) → exposure drops below 100%, and
  the shortfall sits in **cash**.

A few concrete examples:

| Recent volatility | Exposure | What it means |
|---|---|---|
| 12% (calm) | 100% | fully invested |
| 17% (normal) | 88% | 12% moved to cash |
| 25% (turbulent) | 60% | 40% in cash |
| 40% (crisis) | 38% | most of it in cash |

**Where does the de-risked cash go? It stays in cash.** We *tested* moving it into
bonds and into commodities instead — both **lost**, because you'd be de-risking
into a risky asset at the worst possible moment. Cash earns ~4–5% risk-free right
now, and during turbulence that's exactly where you want it.

## The "band" — why it doesn't fidget

Volatility wiggles a little every day. If we retraded every wiggle, we'd rack up
costs. So there's a **±10% no-trade band**: exposure only actually changes when
the new target is more than 10% away from where we already are. Small drifts are
ignored; only meaningful risk changes trigger a trade.

## How often we check

**Every week (Mondays).** Volatility is checked weekly and the exposure dial is
nudged if it's outside the band. This was chosen deliberately: weekly gives you
almost the same protection as checking daily, at a fraction of the trading.

\newpage

# The rebalance calendar (when things actually happen)

Different sleeves move on different clocks. Here's the whole schedule:

| How often | What happens |
|---|---|
| **Weekly (Mon)** | **Equity exposure** check — nudge the invested-vs-cash dial if volatility moved it outside the ±10% band. |
| **Monthly** | **Currency** book — re-rank currencies by interest rate; adjust the long/short list. (Slow-moving; usually small changes.) |
| **Quarterly (~every 63 trading days)** | **Equity holdings** — re-pick the 30 most-liquid names, re-apply the sector cap, reset to equal weight. |
| **Rarely** | **Bonds** — 10% IEF is a *strategic* hold; it only trades to top back up toward 10% as the other sleeves drift. |

So on a typical week, the only thing that moves is a small equity-exposure nudge.
Once a month the currency list gets a small refresh. Once a quarter the equity
names get re-picked. Nothing here is high-frequency — turnover is low by design,
which keeps costs down.

\newpage

# A full worked month

Let's walk through what the system actually does, start to finish.

**Week 1 — Monday (exposure check).** The book's recent volatility reads 17%. The
dial wants exposure = 15% ÷ 17% = **88%**. We were at 100%, and 88% is more than
10% away? No — it's a 12% drop, just outside the band — so we trim to **88%**,
moving 12% of the equity sleeve to cash. On a $40,000 book that's the equity
sleeve going from fully invested to ~$22,400 invested / ~$3,100 cash.

**Weeks 2–3 — Mondays.** Volatility drifts to 16% then 18%. The dial wants 94%
then 83% — both **within 10%** of our current 88%, so **no trade.** The band keeps
us from fidgeting.

**End of the month — currency refresh.** We re-rank currencies by interest rate.
The peso still pays the most, the franc still pays the least, so the long/short
list barely changes — maybe one currency swaps in. A small adjustment.

**Bonds.** Untouched — still sitting at ~10% in IEF, quietly earning interest.

**If it were also quarter-end — equity re-pick.** We'd pull the 30 most-liquid
names fresh, re-apply the 5-per-sector cap, and reset everyone to equal weight.
Any name that fell out of the top-30 by trading volume gets sold; new ones get
bought. This is the only time the *holdings* change.

**Net result for the month:** a couple of small exposure trims, a tiny currency
tweak, no bond trades. Calm, low-turnover, risk kept near target.

\newpage

# What we deliberately DON'T do (and why)

A big part of the strategy is what we *tested and rejected* — because avoiding
things that don't work is as important as doing things that do. Each of these was
built, tested on real data against a "would random noise do this?" benchmark, and
dropped when it didn't earn its keep:

- **Commodities (oil, gold, crops…).** Tested carry and trend strategies on 16
  years of real futures data — both lost money. An optimizer couldn't rescue them
  either. Commodity strategies that worked decades ago have decayed. **Rejected.**
- **Foreign stocks.** Developed markets (Europe, Japan) are ~80% correlated to US
  stocks — nearly the same bet. Emerging-market stocks share a crash-tail with the
  currency sleeve. **Rejected** (adds risk, not diversification).
- **News / sentiment signals.** Tested news-tone, social positioning, and weather
  data as trading signals across stocks, currencies, and commodities. All
  indistinguishable from random noise. Published sentiment is already priced in.
  **Rejected.**
- **Parking de-risked cash in bonds or commodities.** Tested routing the equity
  de-risk slice into bonds/commodities instead of cash. Both *lost* — you'd
  de-risk right into a falling asset (e.g. bonds in 2022). **Cash wins.**

The theme: **simple, liquid, diversified, and only what survives honest testing.**

# How we know it's real (the discipline)

Every result in this strategy was measured the careful way, to avoid fooling
ourselves:

- **No hindsight** ("point-in-time"): backtests only ever used the names/data that
  were *actually available at each past date*, not today's winners looked up in
  reverse.
- **Random-noise tests**: every signal had to beat a distribution of random
  signals. Many promising ideas didn't, and were cut.
- **Real costs**: trading costs and spreads are subtracted, so paper edges that
  die in the real world are caught.
- **Out-of-sample checks**: strategies were fit on older data and tested on newer,
  held-out data — the honest test of whether an edge repeats.

This discipline is *why* the strategy is deliberately small and simple: most
clever additions didn't survive it.

\newpage

# Current status

- **Equity sleeve: live** on Interactive Brokers (the 30-name book, auto-rebalanced
  on the schedule above).
- **3-sleeve design (Equity + Currency + Bonds): built and ready**, verified in a
  dry-run. The equity + bond part is ready to trade live together (both are simple
  long holdings); the currency sleeve is fully specified but is brought live
  deliberately (short-selling currencies needs a careful setup).

# One-page summary

| | **Equity** | **Currency (carry)** | **Bonds** |
|---|---|---|---|
| **Weight** | 63.9% | 26.1% | 10% |
| **Holds** | 30 liquid US stocks, equal-weight | long high-rate / short low-rate FX | IEF (7–10y US Treasuries) |
| **Job** | return engine | diversifier | shock absorber |
| **Rebalance** | holdings quarterly; exposure weekly | monthly | rarely (top-up to 10%) |
| **Safety** | 15% vol-target (de-risk to cash) | dollar-neutral, sized small | small + intermediate duration |
| **Main risk** | broad market fall | sudden "carry crash" | inflation shock (2022-style) |

**The de-risk dial:** exposure = min(100%, 15% ÷ recent volatility), checked
weekly, only trades outside a ±10% band, and the cash it frees up **stays in
cash.**

*Everything here is after realistic costs, uses point-in-time data (no hindsight),
was validated out-of-sample and against random-noise benchmarks, and keeps only
what survived. The result is a simple, broad, risk-managed three-sleeve book.*
