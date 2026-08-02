# autoPortfolio — an automated long-term portfolio research system

A two-stage quantitative system for building a long-term, multi-asset portfolio:

1. **SCREEN (alpha)** — five independent models rank a universe of stocks and
   fuse into one score → *which stocks to hold*.
2. **ALLOCATE (Markowitz / MPT)** — mean-variance optimization decides *how much
   of each* → the efficient frontier, max-Sharpe and min-variance portfolios.

> ⚠️ **Research prototype.** All outputs are currently **in-sample and
> unvalidated**. Do not trade real money until the walk-forward backtester
> exists (see *Roadmap*). The scores show what the models *think*, not proven edge.

---

## Architecture

```
        ┌──────────────────────── SCREEN (screener.py) ───────────────────────┐
        │  A factors.py       value + quality + momentum + value×momentum      │
        │  B sentiment.py     news-headline tone (vaderSentiment)              │
        │  C network_model.py correlation-graph diversifier (low centrality)   │
        │  D regime.py        HMM regime → SETS the weights on A,B,C,E         │
        │  E ml_rank.py       LightGBM LambdaMART learning-to-rank             │
        └───────────────┬─────────── fuse into one SCORE ─────────────────────┘
                        │  top-N stocks
                        ▼
        ┌──────────────────────── ALLOCATE (markowitz.py) ────────────────────┐
        │  data.py → mu (expected returns), Sigma (covariance)                 │
        │  → efficient frontier, max-Sharpe, min-variance (capped weights)     │
        └──────────────────────────────────────────────────────────────────────┘
```

---

## Data — where it comes from, how, and what

**Source:** [Yahoo Finance](https://finance.yahoo.com), accessed through the
`yfinance` Python package. It is free and requires **no API key**, but it is
unofficial and can rate-limit or return partial fundamentals — every module is
written to degrade gracefully when a field is missing.

| What | How (function) | Exactly what we get | Frequency |
|------|----------------|---------------------|-----------|
| **Prices** | `data.download_prices()` → `yf.download(auto_adjust=True)` | Split/dividend-**adjusted close** per ticker | Daily, since 2018 |
| **Returns / risk** | `data.returns_stats()` | Log returns → **μ** = mean×252, **Σ** = cov×252 (annualized) | derived |
| **Fundamentals** | `factors._fundamentals()` → `yf.Ticker(t).info` | trailingPE, priceToBook, EV/EBITDA, ROE, profit margin, debt/equity | snapshot |
| **Momentum** | `factors.momentum_scores()` | 12-1 month price momentum (skips last month) | derived from prices |
| **News** | `sentiment._headline_texts()` → `yf.Ticker(t).news` | Recent **headline titles** → VADER compound tone | latest headlines |
| **Regime** | `regime._market_returns()` → `yf.download("SPY")` | SPY daily returns → 2-state HMM (calm/stress) | daily, since 2015 |
| **ML features** | `ml_rank._features_for()` | momentum (21/63/126/252d), volatility (21/63d), short-term reversal, distance from 52-week high; **target** = forward 21-day return | derived from prices |

**Notes on the data choices**
- *Adjusted* close is used so dividends/splits don't create fake jumps.
- Expected return μ from the historical mean is **naive and noisy** — this is the
  single biggest thing to improve later (shrinkage, Black-Litterman, factor μ).
- Fundamentals are point-in-time snapshots from Yahoo; they are **not**
  survivorship-bias-free and can lag. Fine for research, not for production.

---

## The models

### Stock-picking (the 5-model screener)
| | Model | Idea | Engine (fallback) |
|--|-------|------|-------------------|
| **A** | `factors.py` | Cheap (**value**) + profitable (**quality**) + rising (**momentum**), plus a **value×momentum interaction** (cheap *and* improving) | pandas/numpy |
| **B** | `sentiment.py` | Recent **news tone** shifts lead price | **vaderSentiment** (→ NLTK → keyword lexicon) |
| **C** | `network_model.py` | Build a correlation graph; **low-centrality** names are diversifiers and less arbitraged | networkx eigenvector centrality |
| **D** | `regime.py` | Detect **calm vs stress** market regime; adaptively re-weight the other models (momentum in calm, quality/diversifier in stress) | **hmmlearn GaussianHMM** (→ statsmodels Markov → vol rule) |
| **E** | `ml_rank.py` | **Learning-to-rank**: learn which feature patterns precede outperformance | **LightGBM `LGBMRanker`** LambdaMART (→ sklearn GBR) |

`screener.py` z-scores each signal, weights them by the current regime (D), and
fuses to a single **SCORE**. The top-N becomes the Markowitz universe.

### Allocation (MPT)
`markowitz.py` — from-scratch mean-variance optimization:
`portfolio_return = w·μ`, `variance = w·Σ·w`, `Sharpe = (w·μ − r_f)/√(w·Σ·w)`.
Produces the **min-variance**, **max-Sharpe** portfolios and the **efficient
frontier** (traced from the min-variance point up — the truly *efficient* branch).
`pipeline.py` caps any single name at 25% to force diversification.

---

## Commands

```bash
cd ~/Documents/autoPortfolio
pip install -r requirements.txt           # one-time setup
python -m nltk.downloader vader_lexicon    # one-time (sentiment fallback)

python3 run.py         # MPT efficient frontier on a starter ETF universe
python3 screener.py    # run the 5-model stock ranking, print the table
python3 pipeline.py    # full system: screen -> top-10 -> Markowitz weights
python3 backtester.py       # walk-forward backtest vs benchmark (price signals)
python3 backtester.py --ml  # ...include the ML ranker (slower)
python3 factor_analysis.py  # per-factor IC + quantile analysis (which signals predict?)
python3 universe.py         # build tradeable universe from S&P 1500 -> universe.csv
python3 paper_trader.py     # forward paper-trade one rebalance -> track record
python3 visualize.py   # generate ALL model figures into ./figures/
pytest                 # full offline test suite (113 tests)
pytest -q tests/test_visualize.py   # just the visualization tests
```

---

## Visualizations (`python3 visualize.py` → `figures/`)

| File | Shows |
|------|-------|
| `01_prices.png` | All assets rebased to 100 — relative growth |
| `02_correlation.png` | Return **correlation heatmap** across the universe |
| `03_factors.png` | (A) value / quality / momentum / interaction **z-scores per stock** |
| `04_sentiment.png` | (B) news-sentiment score per stock (green=positive) |
| `05_network.png` | (C) **correlation graph** — node color = cluster, size = centrality; isolated nodes are diversifiers |
| `06_regime.png` | (D) market with **stress-regime shading** + P(stress) timeline |
| `07_ml_importance.png` | (E) which features the ranker relies on |
| `08_screen_scores.png` | **Fused SCORE ranking** + decomposition by model (see who drove each pick) |
| `09_frontier.png` | Markowitz **efficient frontier** with max-Sharpe & min-variance marked |
| `10_weights.png` | Final **max-Sharpe portfolio weights** (pie) |
| `11_backtest.png` | Walk-forward **equity curve vs benchmark** (log) + drawdown |
| `12_factor_ic.png` | Each factor's **IC IR** (annualized signal consistency) |
| `13_factor_quantiles.png` | **Quantile forward returns** per factor (rising bars = predictive) |

The `plot_*` functions in `visualize.py` are **pure** (data in → PNG out), so they
can be reused or embedded elsewhere; `make_all()` is the live orchestrator.

---

## File structure

```
autoPortfolio/
├── README.md            requirements.txt      pytest.ini
│   ── data / allocation ──
├── data.py              prices → mu, Sigma
├── markowitz.py         optimizer: frontier / max-Sharpe / min-variance
├── run.py               MPT demo
│   ── 5-model screener ──
├── factors.py (A)   sentiment.py (B)   network_model.py (C)
├── regime.py (D)    ml_rank.py (E)     utils.py
├── screener.py          fuse A–E → ranked SCORE
├── pipeline.py          screen → allocate (end to end)
├── backtester.py        walk-forward backtest + performance metrics
├── factor_analysis.py   per-factor IC + quantile diagnostics
├── universe.py          build tradeable universe (S&P 1500 + eligibility)
├── paper_trader.py      forward paper-trading loop (SimBroker / IBKR)
├── ibkr.py              IB TWS/Gateway connectivity via official ibapi
├── universe.csv         eligible universe + metrics   (generated)
├── universe_tickers.txt eligible ticker list          (generated)
├── paper_state.json     paper account state           (generated)
├── paper_track_record.csv  forward NAV history        (generated)
├── visualize.py         all model figures → figures/
├── figures/             generated PNGs
└── tests/               79 offline tests (one file per module)
```

---

## Backtesting (walk-forward)

`backtester.py` simulates the strategy through history with **no look-ahead**: at
each monthly rebalance it scores stocks and sets weights using only data up to
that date, then measures the realized return until the next rebalance. It reports
CAGR / annualized vol / Sharpe / max drawdown / turnover vs an equal-weight
benchmark, after transaction costs (default 10 bps).

A strategy is two swappable functions: `score_fn(window)` and
`weight_fn(window, picks)`. Built-ins: `score_momentum`, `score_combined`
(regime-weighted momentum + network + optional ML), `weight_max_sharpe` (capped),
`weight_equal`.

**Point-in-time honesty:** only price-derived signals are backtested (momentum,
network, regime, ML). Value/quality/sentiment are excluded because yfinance only
serves *current* snapshots — using them historically = look-ahead bias.

**Empirical findings (2020→2026 out-of-sample, after 10 bps costs):**
| Strategy | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|
| Combined (momentum+network+regime) | **26.3%** | 23.1% | **0.96** | -27% |
| ...with ML added | 23.2% | 23.5% | 0.82 | -27% |
| Equal-weight benchmark | 18.8% | 19.0% | 0.78 | -34% |

Two lessons the backtest *taught* us: (1) the price-based strategy genuinely beat
buy-and-hold with a shallower drawdown; (2) **adding the ML ranker HURT** results
out-of-sample (Sharpe 0.96 → 0.82) and raised turnover — a textbook example of
in-sample power not surviving out-of-sample. This is exactly why the backtester
exists: to measure, not assume.

## Universe construction (`universe.py`)

Builds the tradeable stock list — separate from, and upstream of, the factor
ranking.
1. **Source:** S&P 500 + 400 + 600 constituents from Wikipedia (~1,500 names —
   the full S&P Composite 1500, large+mid+small).
2. **Eligibility filters** (mechanical, tradability only): price ≥ $3, median
   daily dollar volume ≥ $1M (liquidity/capacity), ≥ 2y history, few data gaps.
3. **Output:** `universe_candidates.csv` (all) and `universe.csv` +
   `universe_tickers.txt` (eligible). Current build: **1,486 eligible — 589 small,
   398 mid, 499 large.**

Keep ALL eligible names (generous) rather than capping by liquidity, which would
crowd out exactly the small/mid-caps Phase 1 targets.

## Forward paper trading (`paper_trader.py`) — the bias-free validation

Runs the strategy FORWARD in a paper account, so results are free of
survivorship bias (you pick from the live universe in real time). Each run:
screens → target weights → rebalances → marks to market → appends a snapshot to
`paper_track_record.csv`. Schedule it monthly (cron / the `/schedule` skill) to
accumulate an honest track record.

- **SimBroker** (default): self-contained paper account in JSON, marks to market
  via yfinance — works today, no setup.
- **IBKRBroker** (`ibkr.py`, IB's official `ibapi`): routes real paper orders to
  Interactive Brokers TWS/Gateway. Sizes orders against your live account NAV.
  `dry_run=True` by default (previews orders without sending).

**IBKR setup (TWS paper):** Configure → API → Settings → check *Enable ActiveX
and Socket Clients*, Socket port **7497**, allow 127.0.0.1. Then:
```bash
python3 ibkr.py 7497                 # read-only connection test (NAV, positions)
python3 paper_trader.py --ibkr       # dry-run: connect, size, PREVIEW orders
python3 paper_trader.py --ibkr --live    # actually transmit paper orders
python3 paper_trader.py --ibkr --gateway # use IB Gateway (port 4002) instead
```
Note: `ib_async` is not used — its `eventkit` dependency collides with macOS
pyobjc's `EventKit` on the case-insensitive filesystem. `ibapi` is dependency-clean.

## Factor analysis — IC & quantiles (`factor_analysis.py`)

Measures whether each price-derived factor actually *predicts* returns, before
trusting it in the strategy.

- **Information Coefficient (IC):** per-date cross-sectional rank correlation
  between a factor and forward returns. `mean_ic` = strength, **`ic_ir`**
  (annualized mean/std) = consistency (the number that matters), `t_stat` =
  significance, `hit_rate` = % of periods pointing the right way.
- **Quantile analysis:** sort stocks into 5 buckets by factor each period,
  average each bucket's forward return. A good factor is **monotonic** (Q1<...<Q5)
  with a positive long-short spread.

Non-overlapping sampling (= horizon) avoids inflated t-stats. Point-in-time safe.

**Finding on our 24 mega-cap universe (2018→2026):** *no* price factor shows
strong, significant, monotonic prediction. The most consistent signal is that
**low volatility HURT** (IC IR ≈ −0.7, t ≈ −2) — i.e. high-beta names led this
bull market; momentum factors were near-zero and non-monotonic. This is expected:
mega-caps are the most efficient, arbitraged names on Earth, so factors are weak
there. It's the strongest quantitative argument for **expanding the universe** to
small/mid-cap or international, where factor premia are larger. It also explains
why the backtest's edge came more from regime timing + diversification + the
Markowitz allocation than from raw factor prediction.

## Testing

`pytest` runs **117 tests, fully offline** — every network call is monkeypatched,
so results are deterministic. Coverage is edge-case focused:

- **Scoring:** zero-variance, all-NaN, single-element, partial-NaN
- **Markowitz:** weights sum to 1, long-only bounds, min-var is lowest vol,
  max-Sharpe beats equal-weight, target-return hit, single asset, shorting
- **Factors:** cheap>expensive, profitable>weak, missing fundamentals → neutral
- **Sentiment:** positive>negative, no-news → neutral, fallback lexicon
- **Network:** edges form for correlated names, **zero-edge graph → neutral**,
  diversifier scores highest
- **Regime:** calm/stress interpolation, probability clipping, short series,
  data-driven stress identification
- **ML:** panel shape, relevance labels, insufficient data → zeros, empty universe
- **Screener/pipeline:** ranking sorted, regime changes weights, weight cap enforced
- **Visualization:** every plot writes a non-empty PNG, incl. single-asset,
  zero-edge, all-zero, empty-probability, NaN inputs

Two real bugs were found *by these tests* and fixed: the efficient frontier was
tracing the inefficient branch, and network centrality returned garbage on a
zero-edge graph.

---

## Roadmap / next steps

- [x] **Walk-forward backtester** — done (`backtester.py`); price-based strategy
      beats the benchmark out-of-sample, ML found to hurt.
- [x] **Per-factor IC / quantile analysis** — done (`factor_analysis.py`);
      showed price factors are weak on mega-caps → motivates a wider universe.
- [x] **Expand the universe** — done (`universe.py`); 1,486 eligible S&P 1500 names.
- [x] **Forward paper-trading loop** — done (`paper_trader.py`); bias-free validation.
- [ ] **Re-run IC analysis + backtest on the new universe** ← next: does the
      small/mid tilt make factors come alive vs the mega-cap baseline?
- [ ] **Schedule** the paper-trading loop monthly (cron / `/schedule`)
- [ ] **Point-in-time fundamentals/news** so value/quality/sentiment can be
      backtested honestly (Norgate/Sharadar)
- [ ] Covariance **shrinkage** (Ledoit-Wolf) + **Black-Litterman** for μ
- [ ] **Threshold rebalancing** (tax/cost aware)
- [ ] Options overlay (covered calls / hedges)
- [ ] **IBKR paper trading** — execution layer, months before real money

---

## Environment caveats
- **xgboost** is intentionally unused (libomp symbol conflict in this Anaconda);
  LightGBM's LambdaMART is the stronger ranker anyway.
- If the HMM regime reports `method='markov'`, run `pip install -U threadpoolctl`
  (a broken build blocks hmmlearn's KMeans init).
- `yfinance` can rate-limit; fundamentals/news may come back empty — the pipeline
  treats missing data as neutral rather than failing.

*Educational research tooling, not financial advice.*
