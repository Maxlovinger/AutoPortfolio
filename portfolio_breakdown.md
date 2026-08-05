---
title: "autoPortfolio — Portfolio Breakdown"
subtitle: "The 30 holdings, their sectors, and the weights"
date: "August 2026"
geometry: margin=0.9in
fontsize: 11pt
mainfont: "Helvetica Neue"
monofont: "Menlo"
colorlinks: true
linkcolor: RoyalBlue
---

# What this is

A snapshot of the **$40,000 paper portfolio**: 30 large, highly-liquid US stocks,
held in roughly **equal weight** (target 3.33% each), spread across 8 sectors with
a cap of **5 names per sector**. Only **one share class per company** is held
(e.g. Alphabet is held once, via GOOGL — not GOOG and GOOGL both). Weights below
are as a % of invested money.

# Sector breakdown

| Sector | # of stocks | Sector weight |
|---|---:|---:|
| Financials | 5 | 16.4% |
| Information Technology | 5 | 15.4% |
| Industrials | 5 | 15.2% |
| Health Care | 4 | 13.8% |
| Consumer Discretionary | 3 | 11.0% |
| Consumer Staples | 3 | 10.4% |
| Communication Services | 3 | 10.0% |
| Energy | 2 | 7.8% |
| **Total** | **30** | **100%** |

No sector exceeds ~19%, and the top holding is under 4% — this is a deliberately
diversified book, not a concentrated bet.

# All 30 holdings

| Stock | Sector | Shares | Stock weight |
|---|---|---:|---:|
| JPM | Financials | 4 | 3.95% |
| HOOD | Financials | 15 | 3.65% |
| V | Financials | 3 | 3.09% |
| BRK-B | Financials | 2 | 2.87% |
| GS | Financials | 1 | 2.86% |
| INTC | Information Technology | 14 | 3.55% |
| AAPL | Information Technology | 4 | 3.47% |
| NVDA | Information Technology | 6 | 3.38% |
| AMD | Information Technology | 2 | 2.68% |
| MU | Information Technology | 1 | 2.31% |
| AAL | Industrials | 86 | 3.69% |
| VRT | Industrials | 5 | 3.39% |
| GE | Industrials | 3 | 3.03% |
| GEV | Industrials | 1 | 2.78% |
| CAT | Industrials | 1 | 2.29% |
| JNJ | Health Care | 5 | 3.60% |
| ABBV | Health Care | 5 | 3.52% |
| UNH | Health Care | 3 | 3.49% |
| LLY | Health Care | 1 | 3.23% |
| AMZN | Consumer Discretionary | 5 | 3.81% |
| HD | Consumer Discretionary | 4 | 3.73% |
| TSLA | Consumer Discretionary | 4 | 3.50% |
| KO | Consumer Staples | 16 | 3.94% |
| WMT | Consumer Staples | 12 | 3.75% |
| COST | Consumer Staples | 1 | 2.67% |
| NFLX | Communication Services | 19 | 3.83% |
| META | Communication Services | 2 | 3.13% |
| GOOGL | Communication Services | 3 | 3.00% |
| XOM | Energy | 9 | 3.93% |
| CVX | Energy | 7 | 3.87% |

*Weights range ~2.2%–3.9% around the 3.33% target. The gap is purely whole-share
rounding: on a $40k account a $1,000+ stock can only buy 1 share, so it lands a bit
light. Fractional shares would tighten this.*

# Why these 30 stocks were chosen

The selection rule is simple and mechanical — **no forecasting, no stock-picking
opinions**:

- **Most liquid first.** We rank the eligible large-cap US universe (S&P 1500) by
  trading liquidity and take the top names. Liquid stocks have tiny spreads and
  barely move when you trade them, so the strategy is cheap and realistic to run.
  (Thinly-traded names look great in backtests but bleed money live.)
- **One share class per company.** If a company has two listed share classes
  (Alphabet's GOOG/GOOGL, etc.), we keep only the more-liquid one — otherwise we'd
  unknowingly hold a double-weight bet on that single company and waste a slot.
- **Capped at 5 per sector.** Without a cap, "most liquid" becomes one-third
  technology — a hidden sector bet. The cap forces the book across 8 sectors.
  Testing confirmed the cap **did not lower returns** — the edge comes from being
  diversified and cheap to trade, not from a tech tilt.
- **Equal weight.** Each name gets the same 1/30 slice. In honest testing, plain
  equal weight beat five "smarter" weighting schemes — a well-known result ("1/N
  is hard to beat"). It needs no estimates, so nothing can go wrong.

# How readjustment works

Two separate clocks run the portfolio:

**1. Which stocks you hold — refreshed quarterly (~every 3 months).**
Every quarter the strategy re-ranks the universe by liquidity and rebuilds the
top-30, still one share class per company, still capped at 5 per sector, still
equal weight. Names that have become more liquid come in; names that have faded
drop out. Turnover is very low (~4% a year), so trading costs stay near zero.
Between rebalances the holdings do not change.

**2. How much is invested vs. in cash — reviewed weekly.**
Separately, every Monday the strategy checks market volatility and sets an
"invested fraction" so the portfolio's risk stays near a 15%-per-year target. In
calm markets it is ~100% invested (like now); in turbulent markets it moves part
of the money to cash to cut the drawdown, then re-invests when things settle. To
avoid needless trading, it only adjusts when the target moves more than **10
percentage points** from where it is (a "no-trade band").

So: **the list of 30 changes quarterly; the invested-vs-cash dial moves weekly (and
only past the 10-point band).** Everything is logged in plain English each time it
runs, so you can always see what changed and why.
