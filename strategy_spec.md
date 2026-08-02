---
title: "autoPortfolio — The Finalized Strategy"
subtitle: "Exactly what it does, and why — in plain language"
author: "Built with Claude Code"
date: "August 2026"
toc: true
toc-depth: 2
geometry: margin=1in
fontsize: 11pt
mainfont: "Helvetica Neue"
monofont: "Menlo"
colorlinks: true
linkcolor: RoyalBlue
---

\newpage

# The strategy in one paragraph

Hold the **30 most-liquid large US companies**, no more than **5 from any one sector**, in **equal amounts**. Re-pick that list once a **quarter**. Separately, every **week**, adjust how much of the portfolio is actually invested (versus sitting in cash) so the portfolio's risk stays near a **15%-per-year volatility target** — but only make that adjustment when it needs to move by more than 10 percentage points, so you're not fiddling constantly. That's the whole thing. It's deliberately simple, because every time we tested something more complicated, it did worse.

\newpage

# 1. Exactly what it does (the rules)

> **Universe:** the 30 most-liquid eligible US stocks (from the S&P 1500), capped at **5 per sector**.
>
> **Weighting:** **equal weight** — each of the 30 gets 1/30 of the invested money.
>
> **Holdings rebalance:** **quarterly** (every ~3 months) — refresh which 30 names you hold.
>
> **Risk overlay:** a **15% annual volatility target**. Invested fraction = 15% divided by the portfolio's recent volatility, capped at 100% (never borrow). When markets are calm you're fully invested; when they get turbulent, part of the money moves to cash.
>
> **Exposure cadence:** review the invested-vs-cash level **weekly**, and only trade it when the target moves more than **10 percentage points** from where you are (a "no-trade band").

Two separate clocks are worth remembering:

- **Which 30 stocks** you own is decided **quarterly**.
- **How much you're invested vs. in cash** is reviewed **weekly** (and only changed past the 10-point band).

\newpage

# 2. Why we made each decision

Each rule below was a *choice*, and each choice beat the alternatives in honest, bias-corrected, cost-included testing. Here's the plain-language reason for each.

## Why the 30 most-liquid stocks?

**Liquidity = you can actually trade it cheaply.** Big, heavily-traded stocks have tiny bid-ask spreads and don't move against you when you buy. Small, thinly-traded stocks look great in backtests but bleed money in real trading. We also tested holding 60, 120, and 200 names — going wider added nothing, and 30 gave the best risk-adjusted return. So: fewer, highly-tradeable names.

## Why cap 5 per sector?

**To avoid secret bets.** Without a cap, the "most liquid" list becomes one-third technology — so you'd unknowingly be betting on tech. Capping at 5 per sector spreads the portfolio across 8+ industries. Crucially, we checked that this cap **did not lower returns** — proof the strategy's edge comes from being diversified and liquid, *not* from a hidden sector bet.

## Why equal weight (and not something cleverer)?

We tested five "smarter" weighting methods (minimum-variance, risk parity, maximum diversification, hierarchical risk parity, inverse-volatility). **Plain equal weight beat all of them** out of sample. This is a famous, repeatable result in finance ("1/N is hard to beat"). The clever methods need to *estimate* how stocks move together, and those estimates are noisy enough that the errors cost more than the cleverness gains. Equal weight needs no estimates, so nothing can go wrong.

## Why rebalance only quarterly?

**Trading costs money, and trading a lot costs a lot.** A high-churn strategy we tested (buying recent losers) traded ~190% of the portfolio every month and lost all its apparent edge to costs. Quarterly rebalancing keeps turnover near 4% a year — almost free — which is a sign of a robust, low-friction design.

## Why the volatility target?

**To control how bad the bad times feel.** Left alone, the portfolio's risk swings wildly — calm in good years, violent in crises. The volatility target keeps risk roughly steady by pulling money into cash when markets get turbulent. In testing it **cut the worst peak-to-trough loss roughly in half** (from about −34% to about −18%) and improved the return-per-unit-of-risk. Importantly, we use it for *risk control* (which is well-supported), not as a magic return booster (which the research shows is unreliable).

## Why review exposure weekly with a 10% no-trade band?

We compared checking exposure daily, weekly, monthly, and only-quarterly:

- **Daily** worked but traded the exposure twice as often (more cost).
- **Monthly** was too slow — drawdowns crept back up.
- **Quarterly-only was useless** — the invested level went stale through a crisis, leaving the drawdown at −34%, as if there were no protection at all.
- **Weekly with a 10-point band** captured essentially all the benefit at half the trading. The band means: don't bother re-trading for small wiggles; only act when the target has really moved.

## The recurring reason behind all of it: simple won, four times

Every time we added sophistication, it lost to the simple version in a fair head-to-head:

1. Equal weight beat five fancy weighting schemes.
2. A simple volatility measure beat a fancier one (EWMA) and beat using the VIX.
3. Volatility targeting beat trend-timing, regime-switching, and portfolio-insurance overlays.
4. A VIX "crisis switch" added nothing on top of the volatility target.

That consistency is *why* the final strategy is simple. It isn't simple by laziness — it's simple because the evidence kept rejecting complexity.

\newpage

# 3. What to expect (and honest caveats)

On honest, survivorship-bias-corrected data with realistic trading costs:

| Metric | Value |
|---|---|
| Return per unit of risk (Sharpe) | ~0.9 |
| Annual return | ~17% |
| Volatility | ~14% |
| Worst peak-to-trough loss | ~ −18% |
| Turnover | very low (~4% / year holdings) |

**Honest caveats — please read these:**

- **These are backtest numbers, not promises.** The real test is the forward paper-trading record we are now accumulating.
- **The test period (2019–2026) was a strong run for large US stocks.** Part of the return rides that wave; a different market could be less kind. Forward testing is exactly what will reveal whether the edge persists.
- **The volatility target protects against slow-building turbulence, not overnight crashes.** In a sudden one-day shock it can't react in time (it uses yesterday's data).
- **~0.9 Sharpe is good, not spectacular.** We deliberately chose an *honest* result over an impressive-looking one. Impressive backtests almost always turn out to be measurement mistakes.

\newpage

# 4. How it runs automatically

The strategy runs itself, and explains itself, every time:

- **A scheduled job** (`auto_rebalance.py`) runs automatically every weekday morning during market hours.
- It figures out what's due: a **quarterly** holdings rebalance, a **weekly** (Monday) exposure check, or nothing.
- It computes the target book and exposure, compares to what's currently held, and sends the needed orders to the **Interactive Brokers paper account**.
- **Every decision is logged in plain English** to `decision_log.md` — which names, why, what the volatility reading was, whether exposure changed and why, and every order. So you can always see *why it did what it did* without reading code.

**Monitoring plan:** glance at the decision log weekly; a monthly check that real fills and costs match expectations; a quarterly review comparing the live record to the backtest. We wrote down in advance what would make us pause the strategy (costs more than double the model, drawdowns behaving like the unprotected version, or any missed run) so we can't rationalize bad results later.

\newpage

# 5. The picture

**Equity curve** — growth of \$1. The orange (volatility-targeted) line is the strategy as run; it's smoother than the un-protected blue line, versus the broad-market benchmark in green.

![Equity curve](/Users/max_lovinger/Documents/autoPortfolio/figures_final/01_equity.png)

**Drawdown** — how far below its recent peak the portfolio is. The volatility-targeted line (orange) stays much shallower, especially through the 2022 decline — that's the risk overlay doing its job.

![Drawdown](/Users/max_lovinger/Documents/autoPortfolio/figures_final/02_drawdown.png)

**Market exposure** — how invested the strategy is over time (1.0 = fully invested, lower = more cash). Fully invested in calm markets, pulling back toward 30–50% in turbulence.

![Market exposure](/Users/max_lovinger/Documents/autoPortfolio/figures_final/05_exposure.png)

**Sector breakdown** — the current book, spread across 8 sectors, capped at 5 names each.

![Sector breakdown](/Users/max_lovinger/Documents/autoPortfolio/figures_final/07_sectors.png)
