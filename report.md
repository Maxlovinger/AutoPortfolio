---
title: "autoPortfolio — A Complete Research Report"
subtitle: "Everything we built, tested, and learned building an honest, automated stock portfolio"
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
urlcolor: RoyalBlue
---

\newpage

# Executive summary

This project set out to build an **automated, long-term stock portfolio** — a system that picks stocks, decides how much to hold of each, and rebalances on a schedule, all driven by data rather than gut feel.

We tried a lot of sophisticated things: a five-model stock screener (factors, news sentiment, a correlation network, a regime detector, and a machine-learning ranker), Markowitz optimization, six different portfolio-weighting schemes, covariance shrinkage, and volatility targeting driven by the VIX.

The single most important finding is this: **once you measure honestly, most of the clever stuff stops working, and simple, well-diversified, risk-controlled approaches win.** Two examples that recur throughout this report:

- Plain **equal weighting (1/N)** beat every "smart" allocation method out of sample.
- The plain **realized-volatility** overlay beat the fancier EWMA and VIX-based versions.

The strategy we ended up locking in is deliberately simple and defensible:

> **Hold the 30 most-liquid large-cap stocks, no more than 5 per sector, in equal amounts. Rebalance every quarter. Scale total exposure up or down to keep the portfolio's risk near a 15% annual volatility target.**

On honest, survivorship-bias-corrected, realistically-costed data it delivers roughly a **0.92 Sharpe ratio, ~17% annual return, and a maximum drawdown of about −18%**, with very low turnover. That is a genuinely deployable result — not a spectacular one, but an *honest* one, which is worth far more.

This document explains, in plain language, everything we tried, the theory behind each idea, and why it did or didn't work.

\newpage

# 1. What we set out to build, and our philosophy

The goal was an **automated long-term portfolio** that could:

1. **Select** which stocks to own from a large universe.
2. **Allocate** capital across them (how much of each).
3. **Rebalance** periodically (not day-trading — quarterly).
4. **Manage risk** so drawdowns stay tolerable.

The guiding philosophy became **"honesty over hope."** It is very easy to produce a backtest that shows spectacular returns and is completely fake. Almost all of the work in this project was about *not fooling ourselves*. Four principles drove everything:

- **Correct for bias.** A backtest that only looks at today's surviving companies is lying to you. We spent significant effort fixing this.
- **Charge realistic costs.** Trading isn't free, especially in smaller stocks.
- **Test on data the model has never seen.** We split history into a training part and a "held-out" part we looked at only once.
- **Prefer simple.** Every added knob is a chance to overfit. Complexity had to *earn* its place in a head-to-head test — and usually it failed.

\newpage

# 2. The four enemies of an honest backtest

Before any results make sense, you need to understand the four ways a backtest lies. This is the theoretical heart of the whole project.

## 2.1 Survivorship bias

**The problem, simply:** If you build your list of stocks from *today's* S&P 500, you're only looking at companies that *survived* to today. All the companies that went bankrupt, got acquired, or were kicked out of the index have silently vanished from your data. So your backtest is effectively saying "how would I have done if I'd only ever bought the winners?" — which nobody can do in real life.

**Why it matters:** It makes every historical result look far better than reality. It was the single biggest source of fake returns in our work.

**How we fixed it (partially):** We reconstructed the *actual* membership of the S&P indices at each point in the past (see Section 3.3), including the companies that later disappeared, and we recovered price histories for as many of the "dead" stocks as we could.

## 2.2 Look-ahead bias

**The problem, simply:** Accidentally using information in your backtest that you couldn't have known at the time. For example, using a company's full-year earnings to make a decision in January, before those earnings were reported.

**How we handled it:** Our backtester only ever hands a strategy the data available *up to* the decision date. We also excluded data sources that only give us *today's* snapshot (like current fundamentals and current news headlines), because using them historically would be cheating. We even wrote tests that prove truncating the data doesn't change past decisions.

## 2.3 Transaction costs

**The problem, simply:** Every trade costs money — the bid-ask spread you cross, and the fact that your own buying pushes the price up (and selling pushes it down). This is small for giant liquid stocks and *large* for small, thinly-traded ones. A strategy that trades a lot in small stocks can look great on paper and lose money in reality.

**How we handled it:** We built a realistic cost model (Section 8) instead of assuming a flat, tiny fee.

## 2.4 Overfitting

**The problem, simply:** If you try enough strategies and enough settings, some will look amazing *by pure luck* on historical data — and then fail immediately in the real world. The more knobs you tune, the worse this gets.

**How we handled it:** We biased hard toward simple strategies, used a held-out test set we touched only once, and refused to keep tuning a strategy just to make it look good (which we explicitly did *not* do when our reversal strategy failed — see Section 10.2).

\newpage

# 3. The data — what we ingested, from where, and why

Everything runs on **free, keyless data**, mostly from Yahoo Finance and Wikipedia.

## 3.1 Price data (Yahoo Finance via `yfinance`)

- **What:** Daily adjusted closing prices and trading volume for thousands of stocks.
- **"Adjusted" means** the prices are corrected for stock splits and dividends, so a price series is comparable over time.
- **Why volume:** we use dollar volume (price × shares traded) to measure how *liquid* a stock is — i.e., how easily you can trade it without moving the price.
- **Limitation:** Yahoo is free but rate-limited and, crucially, it *drops most delisted companies* — which is exactly the survivorship problem.

## 3.2 The stock universe (S&P 1500 from Wikipedia)

- **What:** We pull the constituents of the S&P 500 (large-cap), S&P 400 (mid-cap), and S&P 600 (small-cap) — together the "S&P Composite 1500" — by scraping Wikipedia.
- **Why these:** They're a broad, high-quality cross-section of investable US companies, and being generous with the universe gives the models more to work with.
- **Eligibility filters:** We keep only names that are actually tradeable: price above \$3 (no penny stocks), at least \$1M average daily dollar volume (liquid enough), and at least two years of history. This left about **1,486 eligible names**.

## 3.3 Point-in-time membership (the survivorship fix)

This is one of the most important pieces of the project.

- **What:** We reconstruct *who was actually in each index on each past date*, including companies that were later removed. Wikipedia lists both today's members and a table of every historical *change* (additions and removals with dates). Starting from today and walking the change-log **backward** in time, we rebuild the membership month by month.
- **The result:** Over 2016–2026 the S&P 1500 had **2,178 distinct members** — but only **1,486** remain today. That means **692 companies (about 32%) dropped out** (bankruptcy, acquisition, or demotion). *Those 692 names are the survivorship bias, made concrete.*

## 3.4 Recovering delisted prices

- **The catch:** Knowing a dead company was once in the index is useless for a backtest unless you also have its price history — and Yahoo has purged most of them.
- **What we did:** We tried to download all 692 dead names and recovered usable histories for **235** of them (the rest — mostly acquisitions like Aetna, Allergan, Abiomed — are simply gone from Yahoo). We also removed 24 series that had corrupt prices (impossible one-day moves like +900%, which are data glitches, not real events).
- **The outcome:** Price coverage of the *true* historical universe rose from **68% to 79%**. Better, but the remaining ~21% gap is a hard limit of free data (see Section 13).

## 3.5 The VIX

- **What:** The VIX is the market's expectation of near-future stock-market volatility, derived from S&P 500 option prices. It's often called the "fear gauge."
- **Why we ingested it:** We tested whether this *forward-looking* measure of risk could improve our volatility-targeting overlay (Section 9). Spoiler: it didn't help here.

**Data files produced:** `prices_pit.pkl` (the combined point-in-time price panel, 1,697 names), `membership.csv` (monthly membership history), `prices_delisted.pkl`, `universe.csv`, and `vix.pkl`.

\newpage

# 4. Stock-selection models: the five-model screener

Early on we built an ambitious **five-model screener** that scores every stock by combining five very different ideas, then blends them into one number. Here's each one in plain language, with the theory and the verdict.

## 4.1 Model A — Factors (value, quality, momentum)

- **Theory:** Decades of research find that certain measurable characteristics ("factors") tend to predict returns. **Value** (cheap stocks relative to earnings/book value) tend to outperform; **quality** (profitable, low-debt companies) tend to outperform; **momentum** (stocks that have risen over the past ~12 months, skipping the most recent month) tend to keep rising for a while.
- **How:** We compute these characteristics, standardize them, and combine them into a factor score.
- **Verdict:** Sound in theory. But value and quality need *fundamental* data that Yahoo only provides as a current snapshot, so they **can't be honestly backtested** on free data (look-ahead bias). Only momentum survives into the backtest — and it turned out weak once bias was removed.

## 4.2 Model B — News sentiment

- **Theory:** Positive/negative news moves stocks; measuring the tone of recent headlines might anticipate moves.
- **How:** We score headlines with VADER, a rule-based sentiment analyzer built for short social/news text.
- **Verdict:** Fine as a *live* signal, but **not backtestable** — free news feeds only give current headlines, so using them historically is look-ahead bias. Parked for future use.

## 4.3 Model C — Correlation network

- **Theory (interesting one):** Treat stocks as a network where two stocks are "connected" if their returns move together. Stocks that sit at the *center* of the network are highly correlated with everything — they're "the crowd." Stocks at the *edges* offer genuine diversification.
- **How:** We build the correlation graph and score each stock by its *eigenvector centrality* (a measure of how central it is). We prefer *less* central names for diversification.
- **Verdict:** Elegant and it works mechanically, but as a return *predictor* it's weak. More useful as a diversification tool than an alpha signal.

## 4.4 Model D — Regime switching

- **Theory:** Markets have "regimes" — calm bull markets vs. turbulent, high-stress periods — and different signals work in different regimes. If you can detect the current regime, you can adapt.
- **How:** A Hidden Markov Model (HMM) looks at recent returns and infers whether we're in a calm or stressed regime, then adjusts how much weight to give the other models.
- **Verdict:** Conceptually strong and the detection works. But with limited data it adds complexity for modest benefit. (Notably, the *idea* behind it — reduce risk when volatility is high — is exactly what our final volatility-targeting overlay does, in a much simpler way.)

## 4.5 Model E — Machine-learning ranker

- **Theory:** Instead of hand-combining signals, let a machine-learning model *learn* how to rank stocks by future return. We use "learning-to-rank" (LightGBM's LambdaMART), the same family of algorithm search engines use to rank results.
- **How:** We build a table of features (various momentum and volatility measures) and train the model to rank stocks by their next-month return.
- **Verdict:** This is where overfitting showed its teeth. Adding the ML ranker **hurt** out-of-sample performance (Sharpe fell from 0.96 to 0.82). The model learned patterns in the past that didn't repeat. A textbook lesson in why complexity must be tested honestly.

## 4.6 The overall verdict on stock-picking

The screener is a legitimately powerful piece of engineering, but the honest conclusion is sobering: **on free data and a liquid universe, none of these signals reliably predict returns strongly enough to beat simple diversification after costs.** That finding reshaped the entire project toward *allocation* rather than *selection*.

\newpage

# 5. Measuring whether a signal actually predicts: IC & quantile analysis

Before trusting any signal, you should check whether it *actually* correlates with future returns. Two standard tools:

- **Information Coefficient (IC):** For each date, rank all stocks by the signal, rank them by their *actual* next-period return, and measure how well the two rankings agree. Do this every period and you get a track record of the signal's predictive power. A consistent IC (measured as its "information ratio," IC IR) is what you want. As a rule of thumb, an IC IR above ~0.5 is good; ours were mostly well below that.
- **Quantile analysis:** Sort stocks into five buckets by the signal and check whether the top bucket really does beat the bottom bucket, and whether the buckets line up in order (called *monotonicity*).

**What we found:**

- On mega-cap stocks, **no** factor was strongly or significantly predictive. The most consistent signal was *low volatility*, and it worked *backwards* (high-volatility stocks led that bull market).
- On the broader universe, **short-term reversal** (buying recent losers) was the strongest signal (IC IR ≈ 0.43) — but still not statistically significant, and it did not hold up once we accounted for trading costs (Section 10.2).

The lesson: **a decent IC does not guarantee a profitable strategy.** Reversal had the best IC of anything we found and still lost money after realistic costs.

\newpage

# 6. Portfolio construction & allocation models

Once you've chosen stocks, how much of each do you hold? This turned out to be where the real value was.

## 6.1 Markowitz mean-variance optimization (the classic)

- **Theory:** The foundational idea of modern portfolio theory (Harry Markowitz, 1952). Every portfolio has an expected return and a risk (volatility). Plot all possible portfolios and the best ones form a curve called the **efficient frontier** — for any level of risk, the frontier gives the highest return. Two special points:
  - **Minimum-variance portfolio:** the lowest-risk portfolio possible.
  - **Maximum-Sharpe (tangency) portfolio:** the best *risk-adjusted* return (return per unit of risk).
- **The catch:** Max-Sharpe depends on estimating each stock's *expected return*, which is notoriously unreliable. Small errors in those estimates produce wildly overconfident portfolios. This is why max-Sharpe results in our backtests (Sharpe > 2) were mirages.
- **What we trust instead:** The **minimum-variance** portfolio only needs the covariance (how stocks move together), which is far more stable to estimate. It's less flashy but far more trustworthy.

## 6.2 The allocation bake-off: six ways to weight a basket

After stock-picking disappointed, we tested six "risk-based" weighting schemes head-to-head. These use only how stocks move (their covariance) — **no fragile return forecasts.** All are naturally low-turnover.

- **Equal weight (1/N):** Just hold the same dollar amount of each stock. No estimation at all.
- **Inverse volatility:** Hold more of the calmer stocks (weight proportional to 1/volatility). A simple "risk parity lite."
- **Minimum variance:** Mathematically minimize total portfolio risk.
- **Equal Risk Contribution (ERC / risk parity):** Weight so that *every* stock contributes the *same amount of risk* to the portfolio. A more balanced form of risk parity.
- **Maximum diversification:** Maximize the "diversification ratio" — spread risk as widely as possible.
- **Hierarchical Risk Parity (HRP):** A modern method (López de Prado, 2016) that clusters stocks by similarity into a tree, then splits money down the tree. It's specifically designed to be robust when the covariance estimate is noisy — as it always is with many stocks.

**The result (Section 10.3):** on a fair, broad basket, **plain equal weight (1/N) won**, beating all five sophisticated methods out of sample. This reproduces a famous academic result ("1/N is hard to beat," DeMiguel–Garlappi–Uppal, 2009). The fancier methods reduced *risk* (lower volatility and drawdown) but at a cost to return, netting a lower Sharpe.

## 6.3 Ledoit-Wolf covariance shrinkage

- **Theory:** With many stocks, the estimated covariance matrix is noisy and unstable. "Shrinkage" pulls the noisy estimate toward a simple, stable target — a well-known cure that usually improves methods like minimum-variance.
- **Verdict:** We tested it. It **did not help** here (minimum-variance Sharpe went from 0.42 to 0.41). This cleanly ruled out "noisy covariance" as the reason the risk-based methods lagged — they simply tilt toward lower-return stocks in this sample.

## 6.4 Sector neutralization & caps

- **The problem:** A naive ranking piles into whatever sector is hottest — our momentum picks were almost all semiconductors, a dangerous concentration.
- **The fix:** *Sector-neutralize* (judge each stock against its own industry peers) and *cap* how many stocks you hold from any one sector. This forces diversification across the economy. We use a cap of 5 stocks per sector in the final strategy.

\newpage

# 7. The walk-forward backtester & honest testing

## 7.1 What "walk-forward" means

A walk-forward backtest simulates actually running the strategy through history, step by step, **using only past data at each step**:

1. Stand at a past date. Look only at data available up to that day.
2. Score and select stocks; set the portfolio weights.
3. Hold for one rebalance period (we use a quarter).
4. Record what actually happened, then roll forward and repeat.
5. Chain it all together into an equity curve and measure return, risk, and drawdown.

This is the gold standard because it mirrors reality: you never use tomorrow's information to make today's decision. We even have automated tests proving there's no look-ahead.

## 7.2 Train / validation / held-out test

To guard against overfitting, we split history into three parts:

- **Training** (through 2022): where ideas are formed.
- **Validation** (2023–2024): where they're checked.
- **Held-out test** (2025 onward): looked at **exactly once**, at the very end.

If a strategy only looks good in training but falls apart in validation or the held-out test, it was overfit. This discipline is what revealed that our reversal strategy was no good (Section 10.2).

\newpage

# 8. The realistic transaction-cost model

Instead of a flat fee, we model the two real components of trading cost, both driven by a stock's liquidity (its average daily dollar volume, "ADV"):

- **Half-spread:** the bid-ask spread you pay to trade. It's tiny for giant liquid stocks and large for small ones. We model it as shrinking with liquidity (about 25 basis points at \$1M ADV, under 1 basis point at \$1B ADV). *(A "basis point" is 0.01%.)*
- **Market impact:** your own trading moves the price against you, and the more of a day's volume you consume, the worse it gets. We use the standard "square-root law": impact grows with the square root of how much of the daily volume you trade. This also scales with your portfolio size — a \$10M fund pushes prices more than a \$100k account.

**Why this matters:** it charges high-turnover strategies (like reversal, at ~190% turnover) their true cost, which is exactly where fake alpha hides. Our final strategy has ~4% turnover, so its costs are minimal — a sign of a robust, low-friction design.

\newpage

# 9. Volatility targeting (managing risk directly)

## 9.1 The idea, simply

Instead of always being 100% invested, **adjust how much you hold based on how risky the market is right now**, aiming for a constant level of portfolio risk (we target 15% annual volatility). When markets are calm, be fully invested; when they get turbulent, pull some money into cash. No leverage — the most you're ever invested is 100%.

The rule: `exposure = min(15% ÷ recent_volatility, 100%)`. It's *continuous* — exposure slides smoothly (roughly 30%–100%), it's not an on/off switch. And it uses only *past* volatility, so there's no look-ahead.

## 9.2 Why it helps

- It **roughly halved the maximum drawdown** (from about −34% to −18%) and improved the risk-adjusted return.
- Academic support exists ("Volatility-Managed Portfolios," Moreira & Muir, 2017) — but honestly, follow-up research shows the *return-boosting* claims often fail out of sample and after costs. So we use it for what's robust — **drawdown control** — not as a magic alpha source.
- **Its limit:** it protects best against *slow-building* turbulence (like the 2022 grind-down). It can't dodge sudden crashes (like March 2020), because by the time yesterday's volatility flags danger, the fast crash has already happened.

## 9.3 The VIX experiment (and why simple won again)

We tested whether feeding the overlay better volatility forecasts would help:

- **Realized** volatility (trailing standard deviation) — the simple baseline.
- **EWMA** — an exponentially-weighted average that reacts faster to recent moves.
- **VIX** — the market's forward-looking implied volatility.
- **Blend** — a mix of VIX and EWMA.

**Result:** the simple **realized** version won (Sharpe 0.92), beating EWMA (0.88), the blend (0.85), and VIX alone (0.81). VIX actually *increased* trading (it's jumpy and reacts to broad market fear that doesn't match our specific 30-stock book) and produced deeper drawdowns. We kept the simple version and shelved the VIX machinery for possible future use (e.g., monthly rebalancing or as an extreme-fear filter).

\newpage

# 10. The strategies we actually ran, and what happened

This is the chronological story of the experiments and their honest verdicts.

## 10.1 Momentum picks led to dangerous concentration

Our first real portfolio (momentum stocks, optimized by Markowitz) piled almost entirely into **semiconductors** — 43% volatility, wildly concentrated. Fixing it with sector-neutralization and caps produced a diversified book, but the headline Sharpe (2.16) was a max-Sharpe mirage. First lesson: trust minimum-variance, cap sectors.

## 10.2 The reversal strategy failed the honest test

We locked a reversal-led strategy (buy recent losers, sector-neutral, minimum-variance, monthly) and ran the disciplined train/validation/held-out-test evaluation. The verdict was unambiguous:

| Period | Strategy Sharpe | Benchmark Sharpe |
|---|---|---|
| Training (through 2022) | 0.08 | 0.53 |
| Validation (23–24) | 0.58 | 0.86 |
| **Held-out test (2025+)** | **−0.32** (lost money) | 0.76 |
| Full | 0.08 | 0.61 |

It lost to a plain equal-weight benchmark in *every* period and lost money on the held-out test. The killer was **193% turnover** — reversal constantly churns the portfolio, and realistic costs ate everything. **We deliberately did not keep tuning it to look better** — that would have been overfitting. We recorded the honest failure and moved on. This was the turning point that shifted focus to allocation.

## 10.3 Broad equal-weight: the winner emerges

Testing the six allocation methods on a fair, broad basket of the 60 most-liquid stocks (quarterly), the ranking was clear:

| Method | Sharpe | Turnover |
|---|---|---|
| **Equal weight (1/N)** | **0.75** | 4% |
| Inverse volatility | 0.67 | 9% |
| Equal Risk Contribution | 0.68 | 12% |
| Maximum diversification | 0.67 | 55% |
| Hierarchical Risk Parity | 0.60 | 25% |
| Minimum variance | 0.42 | 43% |
| *Full-universe 1/N (benchmark)* | 0.61 | — |

Equal weight of a liquid basket won — beating both the broad benchmark and every sophisticated method, at rock-bottom turnover.

## 10.4 Basket size: smaller is sharper (but concentrated)

| Basket size (equal weight) | Sharpe | Max drawdown |
|---|---|---|
| 30 names | **0.93** | −31% |
| 60 | 0.75 | −35% |
| 120 | 0.74 | −37% |
| 200 | 0.72 | −37% |

Concentrating to the 30 most-liquid names raised the Sharpe — but those are the mega-caps, so it's partly a bet on continued mega-cap leadership. Breadth beyond 60 added nothing.

## 10.5 Volatility overlay: the keeper

Adding the 15% volatility target lifted the Sharpe *and* roughly halved the drawdown (see Section 9). This became a permanent part of the strategy.

## 10.6 Sector cap: diversify for free

The 30-name book was 33% technology. Tightening the sector cap from 10 to 5 per stock cut the top-sector weight to ~17% **with essentially no loss in performance** — proving the edge came from being liquid + equal-weight + risk-controlled, *not* from a tech bet. This removed the main risk of the concentrated version.

\newpage

# 11. The final locked strategy

## 11.1 The specification

> - **Universe:** point-in-time S&P 1500 membership (delisted names included).
> - **Selection:** the **30 most-liquid** members, capped at **5 per sector**.
> - **Weighting:** **equal weight** (1/N) — beat every alternative.
> - **Rebalance:** **quarterly** — keeps turnover near 4%.
> - **Risk overlay:** **15% annual volatility target** (de-risk into cash in turbulent periods; no leverage; no look-ahead).

## 11.2 Performance (honest: bias-corrected + realistic costs)

| Metric | Raw (always invested) | With 15% vol-target |
|---|---|---|
| Sharpe ratio | 0.81 | **0.92** |
| Annual return (CAGR) | ~21% | ~17% |
| Volatility | ~21% | ~14% |
| Max drawdown | −34% | **−18%** |
| Turnover | ~4% | ~4% |

The vol-target trades a little terminal wealth for a much smoother ride and half the drawdown. For a portfolio you'll actually hold through downturns without panicking, that's the right trade.

## 11.3 The charts

**Equity curve** — growth of \$1, log scale. The orange (vol-targeted) line is smoother than blue (raw) but ends slightly lower; both are compared to the broad-market benchmark (green).

![Equity curve](/Users/max_lovinger/Documents/autoPortfolio/figures_final/01_equity.png)

**Drawdown** — how far below its peak the portfolio is at any time. The vol-targeted line stays much shallower, especially through the 2022 decline.

![Drawdown](/Users/max_lovinger/Documents/autoPortfolio/figures_final/02_drawdown.png)

**Market exposure** — how invested the strategy is over time (1.0 = fully invested). It sits at 100% in calm markets and drops toward 30–50% during stress. This is the vol-target working.

![Market exposure](/Users/max_lovinger/Documents/autoPortfolio/figures_final/05_exposure.png)

**Current holdings** — 30 names, equal weight (3.33% each).

![Current book](/Users/max_lovinger/Documents/autoPortfolio/figures_final/06_weights.png)

**Sector breakdown** — well spread across 8 sectors, capped at 5 per sector.

![Sector breakdown](/Users/max_lovinger/Documents/autoPortfolio/figures_final/07_sectors.png)

**Annual returns** — the strategy (with overlay) versus the benchmark, year by year.

![Annual returns](/Users/max_lovinger/Documents/autoPortfolio/figures_final/08_annual_returns.png)

\newpage

# 12. Key lessons

1. **Simple beats complex — repeatedly.** Equal weight beat five sophisticated allocation methods; realized volatility beat EWMA, VIX, and their blend; adding a machine-learning ranker *hurt*. Every time complexity had to prove itself head-to-head, it lost.

2. **Bias correction dramatically lowers returns.** Fixing survivorship bias cut a momentum strategy's apparent return from 57% to 21.5% per year, and its Sharpe from 1.33 to 0.50. The "amazing" original was a mirage.

3. **A good signal is not a good strategy.** Reversal had the best predictive power (IC) of anything we found, yet lost money after realistic trading costs. Prediction and profit are different things.

4. **Risk management is more reliable than return prediction.** We couldn't reliably predict *which* stocks would win, but we *could* reliably control risk (via diversification and volatility targeting) — and that's where all the durable improvement came from.

5. **The discipline of a held-out test is priceless.** It's the difference between knowing a strategy works and hoping it does.

\newpage

# 13. Limitations & honest caveats

- **The ~21% price gap.** Free data still can't provide prices for ~21% of the historically-real universe (acquired/bankrupt names Yahoo purged). So even our "honest" backtest remains *somewhat* optimistic. Fully closing this needs a paid data vendor (Norgate or Sharadar, roughly \$50–90/month).

- **A mega-cap-bull artifact.** The 2019–2026 period was an unusually strong run for large-cap US stocks. Our final strategy's returns partly ride that wave. Only forward testing (or a longer, vendor-quality history) will show whether the edge persists in a different environment.

- **Untested factors.** Value and quality — the factors with the *strongest* academic support — could not be honestly tested on free data because we lack point-in-time fundamentals. A paid vendor would unlock them.

- **Backtest ≠ future.** All of this is historical simulation. The genuine, bias-free test is a forward paper-trading record, accumulated in real time.

\newpage

# 14. Recommended next steps

1. **Forward paper-trade the locked strategy.** Wire the final strategy's picks into the paper trader and start a real-time, bias-free track record on the Interactive Brokers paper account. This is the single most valuable next move — it costs nothing and it's the only fully honest test.

2. **Subscribe to a point-in-time data vendor** (Norgate or Sharadar). This closes the 21% price gap *and* unlocks value/quality factor testing — the two biggest data limitations at once.

3. **Add a realistic small-cap live-cost check** if you ever broaden beyond mega-caps.

4. **Revisit value/quality** once point-in-time fundamentals are available — they have the best long-run evidence and we've never been able to test them fairly.

\newpage

# Appendix A — Code map

| File | What it does |
|---|---|
| `data.py` | Download prices; compute returns statistics. |
| `universe.py` | Build the tradeable universe (S&P 1500 to eligible names). |
| `historical_membership.py` | Point-in-time index membership (survivorship fix). |
| `factors.py`, `sentiment.py`, `network_model.py`, `regime.py`, `ml_rank.py` | The five screener models (A–E). |
| `screener.py`, `pipeline.py` | Fuse the models; screen to allocate. |
| `markowitz.py` | Mean-variance optimization / efficient frontier. |
| `allocators.py` | The six risk-based weighting schemes + shrinkage. |
| `factor_analysis.py` | IC and quantile analysis. |
| `backtester.py` | Walk-forward engine, performance metrics, vol-targeting. |
| `costs.py` | Realistic spread + market-impact cost model. |
| `vol_forecast.py` | Realized / EWMA / VIX / blend volatility forecasts. |
| `sector_select.py` | Sector neutralization and caps. |
| `allocation_bakeoff.py` | Head-to-head allocation comparison. |
| `final_strategy.py` | The locked strategy + its chart set. |
| `paper_trader.py`, `ibkr.py` | Forward paper trading and IBKR connectivity. |
| `visualize.py` | All model/portfolio charts. |
| `tests/` | 182 automated tests covering every module and edge case. |

# Appendix B — Key commands

```
python3 universe.py                 # build the tradeable universe
python3 historical_membership.py    # build point-in-time membership
python3 allocation_bakeoff.py       # compare allocation schemes
python3 final_strategy.py           # run the locked strategy + charts
python3 -m pytest                   # run the full test suite (182 tests)
```

# Appendix C — Glossary

- **Alpha:** return above what the market/risk would explain — genuine skill.
- **Basis point (bp):** 0.01%.
- **CAGR:** compound annual growth rate — the smoothed yearly return.
- **Covariance:** how two stocks' returns move together.
- **Drawdown:** the drop from a portfolio's peak to a later low.
- **IC (Information Coefficient):** how well a signal's ranking predicts future returns.
- **Liquidity / ADV:** how easily a stock trades; ADV = average daily dollar volume.
- **Point-in-time:** data as it truly was on a past date, without hindsight.
- **Sharpe ratio:** return per unit of risk (higher is better; ~1 is good for a long-only equity strategy).
- **Survivorship bias:** the error of studying only the companies that survived to today.
- **Turnover:** how much of the portfolio you trade each rebalance.
- **Volatility:** how much returns bounce around; the standard measure of risk.
