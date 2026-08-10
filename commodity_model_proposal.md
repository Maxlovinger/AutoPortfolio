# Commodity Model — Proposal (Tier 2)

*Status: proposal. Tier-1 gate PASSED and the Tier-1.5 overlay screen is DONE (§5 — only fundamental surprises survived); the price engine (Tier-2) is not yet built. Kept separate from the equity, FX-carry, and (rejected) foreign-equity sleeves; shares infrastructure only: walk-forward validation, realistic costs, the `commodity_signals` null-test harness, the HMM regime overlay, markowitz.py, and the IBKR execution path.*

*Date: 2026-08-09*

---

## 0. Why this one is worth building (what Tier 1 showed)

The Tier-1 ETF-proxy gate (`commodity_gate.py`, 2006–2026, 108-mo overlap) gave the first **PASS** of any diversifier we've tested:

- The constructed **trend book** was **negatively correlated to the equity book (−0.09)** and, crucially, returned **+1.09%/month in the equity book's worst 20% of months** (when equities averaged −4.76%) — genuine **crisis alpha**, the property foreign equity *lacked* (it co-crashed).
- **Markowitz** gave a passive basket ~15% and the trend book ~16% weight, lifting combined Sharpe 0.89 → 0.91.
- **But** passive commodities were ~0 standalone Sharpe (contango bleed), and the crude trend proxy was ~0 Sharpe too.

The key asymmetry vs foreign equity: **Tier 1 *understates* this strategy.** The proxy was an unlevered, equal-weight `sign(12-month return)` on 6 ETFs — no vol-targeting, no carry leg, and ETFs that bleed roll yield. The literature's actual construction (vol-targeted, carry + trend, 20+ markets) is materially better. So Tier 2 is about **capturing the real return** that sits on top of the crisis alpha we already measured — not about re-proving the diversification.

## 1. Thesis

Commodity futures returns come from the **shape of the futures curve (roll yield / carry)** and from **persistent trends**, not from spot prices rising (real spot ≈ flat over decades — Erb & Harvey 2006). Two documented, decades-long premia:

- **Term-structure / carry:** backwardated commodities (near > far) pay a positive roll yield; contangoed ones bleed. Long backwardated / short contangoed has paid a premium tied to the *theory of storage* — low inventories → backwardation → higher expected return (Gorton, Hayashi & Rouwenhorst 2013). **This is the FX carry trade in a different market** — we already own the machinery.
- **Trend / time-series momentum:** commodities trend strongly and provide **"crisis alpha"** (Moskowitz, Ooi & Pedersen 2012) — the managed-futures diversification benefit.

We do **not** forecast commodity prices. We build a **cross-sectional + time-series systematic book** on carry and trend, vol-targeted, with a sentiment/positioning overlay, sized as a **tail-hedging diversifier** (~10–15%), not a return engine.

## 2. Universe

~24 liquid, exchange-traded futures across four sectors (breadth is the diversification):

- **Energy:** WTI crude (CL), Brent (BZ), RBOB gasoline (RB), heating oil (HO), natural gas (NG)
- **Metals — precious:** gold (GC), silver (SI), platinum (PL), palladium (PA)
- **Metals — base:** copper (HG); LME aluminum/zinc/nickel if data allows
- **Grains/oilseeds:** corn (ZC), wheat (ZW), soybeans (ZS), soybean oil (ZL), soybean meal (ZM)
- **Softs:** sugar (SB), coffee (KC), cotton (CT), cocoa (CC)
- **Livestock:** live cattle (LE), lean hogs (HE)

Gate each contract behind a **liquidity + cost filter** (volume, open interest, spread) exactly like the EM currency sleeve. Start with the ~15 most liquid; add softs/livestock only if they survive costs.

## 3. Instruments & return accounting

Trade **futures**, never physical. Two data objects per market:

1. **Roll-adjusted continuous series** (back-adjusted) — for the **trend** signal and for return realization.
2. **The front two contract prices** (or the full curve) — for the **carry** signal (the slope) and to model roll cost.

Per-period return of a long position (excess return, futures are self-funding):

    R_i  =  spot/price return_i (from the continuous series)  +  roll return_i

where the roll return is the curve-slope effect. Carry (annualized) is measured from the two nearest contracts:

    carry_i  =  (P_near − P_far) / P_far  ×  (365 / days_between)

Positive carry = backwardation = long candidate; negative = contango = short candidate. (Identical shape to the FX `carry_i = rate_i − rate_USD` we already rank on.)

## 4. Signals

**A. Carry / term structure (the anchor)**
- Rank all markets by annualized front-curve slope; long top sleeve, short bottom, risk-weighted. Reuses the exact cross-sectional ranker from `fx/backtest_carry.py`.

**B. Trend / time-series momentum (the crisis-alpha engine)**
- Per market: blended 3/6/12-month trend sign (or a vol-normalized trend score), lagged (no lookahead). Directional per market (a market can be long *or* short regardless of the cross-section).

**C. Value (mean-reversion, secondary)**
- Deviation of current real price from a 5-year average (Asness, Moskowitz & Pedersen 2013). Long cheap / short expensive. Weak alone; diversifies trend.

**D. Positioning — hedging pressure (from COT)** — *screened out (§5).* Commitment-of-Traders crowding (Hong & Yogo 2012) was tested on the proxies and did **not** beat the null (z 0.11–0.66); not in the base build.

**E. Overlays (section 5 — screened).** Of the four candidate overlays, only **EIA/USDA fundamental surprises** survived the null; news tone, COT positioning, and weather were rejected on the proxies.

The base book is **carry + trend**, vol-targeted (the two the papers agree on). Value is a secondary tilt still to be tested; the only *external-data* overlay carried forward is the **fundamental-surprise** signal — everything else was measured and dropped.

## 5. Sentiment/overlay layer — SCREENED (Tier-1.5, results in)

This section is no longer a proposal — the four candidate overlays were **built and tested** ahead of the price engine (`commodity_signals.py` harness + `commodity_factors.py` builders; 30 offline tests). Each factor was scored against **next-month commodity returns** on the ETF proxies, with strict no-lookahead and a **300-permutation null** (the same bar that rejected equity/FX sentiment). Result table (2026-08-09, ETF proxies, `n_null=300`):

| Overlay | test | IC | **z vs null** | verdict |
|---|---|---|---|---|
| **EIA inventory surprise** (oil, gas) | time-series | 0.100 | **2.23** | **KEEP** ✅ |
| Weather — natgas degree-days (NOAA) | time-series | 0.082 | 1.08 | drop |
| COT positioning | both | 0.014 | 0.11–0.66 | drop |
| **News tone (GDELT + FinBERT)** | both | 0.010 | **0.14–0.30** | **drop — noise** |

**The one survivor — fundamental surprises.** EIA weekly petroleum/natgas storage *surprise* (actual change minus a prior-years-only seasonal expectation, sign-flipped so a bigger-than-normal draw is bullish) cleared the null on IC. It stays in the design, extended to **USDA WASDE/NASS** grain-stocks surprises by the same construction. Honest size limits: z = 2.23 is *just* over the bar, on only **2 markets**, ~1 of 4 factors tested (deflate for multiple testing), and the tradable tilt is weaker (Sharpe z 1.55). So it enters as a **modest tilt**, not a pillar — and must re-clear the null on the real futures book.

**The three rejects — measured, not assumed.**
- **News tone (the original explicit ask) is noise here.** Full 13-market GDELT panel, IC ≈ 0.01, z ≈ 0.1–0.3, *negative* tilt Sharpe — and completing the panel from 8→13 markets pushed it *closer* to zero. This is the **4th confirmed "published sentiment is noise"** in the project (equities, FX, earnings, now commodity news), even though commodities were the *fairest* test (discrete, news-legible events). Published, commoditized tone is already priced. FinBERT/GDELT infra kept, but the tone tilt is out.
- **COT positioning** and **NOAA weather** likewise failed the null on the proxies. Not built into the base design.

**How the surviving overlay enters:** the fundamental-surprise signal is a **water-filling-capped `(1 + λ·z)` tilt** on the carry+trend weights (the `equity_sentiment.py` mechanism), applied only to the energy (and later ag) markets it scores. Every overlay — including this one — must **re-beat the null on the real futures book** before it's trusted; the ETF-proxy pass is necessary, not sufficient. (Rejected overlays may be *re-tested* on the futures book but are not in the base build.)

## 6. Portfolio construction

- **Per-market vol-targeting:** scale each position by 1/σ̂ (recent realized vol) so no single market dominates — the managed-futures standard.
- **Signal blend:** combine carry + trend (+ value/positioning/sentiment tilts) into one score per market; long/short.
- **Portfolio vol-target** to a fixed annualized risk (e.g. 10–12%), consistent with sizing this as a satellite.
- **Sector risk caps** (energy/metals/ags/softs) so the book isn't a levered oil bet — analogous to the equity sector cap.
- **Rebalance** monthly (carry is slow); trend checked more often only if it survives turnover costs (the FX lesson: weekly rarely beats monthly for slow signals).

## 7. Risk overlay

- **Momentum crashes** are the trend book's tail (Daniel & Moskowitz) — sharp reversals whipsaw trend. Vol-targeting plus a reversal/vol-spike damper mitigates.
- Reuse the **HMM crash overlay** (`fx/regime.py`) to cut gross exposure in cross-asset stress.
- Report **skew and worst-month** prominently, as with carry.

## 8. Costs — baked in from day one

Futures costs differ from equities/FX and must be modeled explicitly:

- **Commission + exchange/clearing fees** per contract (IBKR schedule).
- **Slippage / half-spread** per market, scaled by participation vs volume (Almgren-style, reuse `costs.py` adapted to contract volume).
- **Roll cost:** every roll crosses a spread — the dominant recurring cost for a monthly book. Model per-market roll spread.
- Softs/livestock and back-month contracts are the expensive, thin corner — gate them behind cost survival, like EM currencies.

## 9. Data sources needed

**Futures prices (the crux — need BOTH continuous and the curve):**
- **Primary — Interactive Brokers (`reqHistoricalData`).** We already run TWS (`ibkr.py`). IBKR gives per-expiry contract history → we can build the continuous series (trend) **and** read front/second contracts (carry/roll) **and** it's the execution venue (consistent fills). Cost: per-exchange market-data subscriptions. **Recommended default.**
- **Deep-history backfill (IBKR history can be short):** a roll-adjusted continuous-futures vendor — **CSI Data**, **Norgate Data**, or **Databento** (pay-as-you-go, full CME). Nasdaq Data Link's free continuous futures (the old "Wiki/CHRIS") is **deprecated**; its Stevens Continuous Futures is now paid.
- **Free / prototyping only:** yfinance front-month continuous (`CL=F`, `GC=F`, `ZC=F`, …) — usable for the **trend** signal, but **no curve** (can't compute carry) and roll gaps. Good enough to extend Tier-1, not to build Tier-2 carry.

**Term structure / carry:** front two contract prices per market (from IBKR per-contract, or a curve vendor).

**Positioning:** **CFTC Commitments of Traders** — free, weekly (Socrata API / `cot_reports` package).

**Fundamentals:** **EIA** (energy, free API), **USDA NASS/WASDE** (ags, free QuickStats API), **NOAA** (weather, free), COMEX/LME warehouse stocks (metals; some paid).

**Sentiment/news:** GDELT (`news_gdelt.py`, free, built), **Alpha Vantage NEWS_SENTIMENT** (key in `.env`), FinBERT (`news_sentiment.py`, built).

**Consensus** for fundamental surprises (EIA/WASDE expectations) is the expensive/hard part — same problem as FX macro consensus; can start with a naive survey or lagged-model expectation and upgrade later.

**Data readiness (verified 2026-08-09):** `.env` keys all present and valid — `FRED_API`, `ALPHA_VANTAGE_API`, `EIA_API`, `QUICKSTATS_API`, `NOAA_API` (values are single-quoted; both dotenv and the `fx/data.py` fallback strip quotes, so this is fine). Keyless sources reachable: **CFTC COT** ✅, **GDELT** ✅ (429 only from datacenter IPs — works locally), **US Drought Monitor** ✅. **FinBERT** local (built). **IBKR** available (TWS open) for the futures prices — the one still-outstanding *price* feed; the **entire news/fundamentals/weather side is now in hand.**

**NOAA — in hand.** `NOAA_API` token added and verified against **NOAA CDO v2** (GHCND daily summaries; token passed in the `token:` HTTP **header**, not a query param). This gives historical daily temperatures → heating/cooling **degree days** (natgas & crop-stress features), complementing EIA's own degree-day series and the keyless Drought Monitor. Note CDO rate limits (5 req/s, 10k/day) and station-by-station queries → cache to a pickle like the other feeds.

**One history caveat for the news test:** Alpha Vantage `NEWS_SENTIMENT` only reaches back to ~2022 (shallow for backtesting) and is rate-limited (25/day free). **GDELT + FinBERT** (2015+) is the *backtestable* tone source — use AV as a live/recent cross-check, GDELT+FinBERT as the historical signal, exactly as in the equity sentiment work.

## 10. Validation

Same discipline that has rejected most of what we've tried — this is the gate that decides if it's real:

- **Walk-forward, purged/embargoed** around report dates (no lookahead across EIA/WASDE/COT release boundaries).
- **Benchmarks to beat:** (a) plain carry-only book, (b) plain trend-only book, (c) the Tier-1 ETF proxy. Each added signal (value, positioning, sentiment) must beat the carry+trend base **out-of-sample, after costs**.
- **Random-signal / random-tone nulls** for every discretionary-looking signal (the bar that killed equity & FX sentiment).
- **Deflated Sharpe** for the number of configs tried.
- **Re-run the diversification gate** with the *real* book (not the ETF proxy) against equity + carry: confirm the crisis alpha and the ~15% Markowitz weight survive with true futures returns and costs. Report skew/worst-month, not just Sharpe.

## 11. Build order

- **0. Tier-1 gate — DONE** (`commodity_gate.py`): PASS on crisis-alpha diversification (§0).
- **0.5. Tier-1.5 overlay screen — DONE** (`commodity_signals.py` + `commodity_factors.py`, §5): only **EIA/USDA fundamental surprises** survived the null; news/COT/weather dropped.
1. **Ingest IBKR (or vendor) futures** — continuous + front/second contract — for the ~15 most-liquid markets; build the data layer (`commodity_futures.py`, mirroring `fx/data.py`). *← the next step (see below).*
2. Backtest **plain carry** (curve slope rank) with realistic roll costs → benchmark bar #1.
3. Backtest **plain trend** (vol-normalized TS-momentum), vol-targeted → benchmark bar #2. Confirm the crisis alpha on real data (not the ETF proxy).
4. Combine **carry + trend**, per-market and portfolio vol-target, sector caps → the base book.
5. Add the **value** tilt (5yr real-price reversion); keep only if it beats the base OOS after costs.
6. Add the surviving **EIA/USDA fundamental-surprise overlay** (§5) via the existing harness (`commodity_signals.evaluate_signal`) — re-clear the null on the *futures* returns, then wire as a capped `(1+λ·z)` tilt. (News/COT/weather already screened out; re-test only if cheap.)
7. Add the **HMM/vol crash overlay**; verify it improves skew & drawdown.
8. Re-run the **diversification/Markowitz gate** with the real book; decide final sizing (≤ the MV weight, skew-aware).

## 12. Honest expectation

- **Carry + trend, vol-targeted:** a real, defensible premium; modest Sharpe (~0.4–0.7 unlevered), with genuine **crisis-alpha** diversification — the reason to run it. Not a high-Sharpe printer.
- **Passive long commodities:** ~0 Sharpe (contango) — not worth holding on its own.
- **Overlays (now screened, §5):** only **fundamental surprises (EIA/USDA)** survived the null — a modest tilt, not a pillar. **News tone, COT positioning, and weather were measured and dropped** (news IC ≈ 0, the 4th "published sentiment is noise" in the project). So the sleeve is essentially **carry + trend + a small energy/ag fundamental tilt** — cleaner than first proposed.
- **Momentum crashes + financialization (Tang & Xiong 2012)** mean the diversification is weaker than pre-2004 backtests and comes with its own tail — size accordingly.
- **Success = a modest, better-behaved sleeve that adds crash protection and a real (if small) Markowitz weight after costs** — the diversifier the portfolio has been missing, not an alpha windfall.

## 13. Idea to test (open): redeploy de-risked equity exposure into commodities

**The idea (user, 2026-08-09):** the equity book's 15% vol-target overlay moves a fraction `(1 − e_t^eq)` of the book **to cash** when equity volatility spikes. Instead of parking that in cash, **route it into the commodity (trend) sleeve.**

**Why it's attractive — a timing match.** The equity overlay de-risks *exactly* in turbulent/stress months, and those are the months where the Tier-1 trend book showed **crisis alpha** (+1.09%/mo when equities averaged −4.76%). So the capital freed up during equity stress would flow into the one sleeve that has historically *risen* then. It's also capital-efficient: the sidelined slice earns the diversifier's expected-positive stress return instead of ~cash.

**Why it might not work — the honest failure modes (what the test must rule out):**

1. **Whipsaw / timing mismatch.** Equity-vol spikes and trend's crisis alpha are correlated but *not the same event*. A sharp spike-and-reverse selloff (Feb-2018 "volmageddon", Mar-2020) de-risks equities near the bottom **and** whipsaws trend-following (momentum needs a *sustained* move). You could lose on both legs: sell equities low, shovel into trend, get whipsawed. The +1.09% is an average over few events (low power) — the variance is large.
2. **It undoes the point of the overlay.** Cash is a *guaranteed* de-risk. Replacing it with an active sleeve (with its own vol and its own momentum-crash tail) re-introduces risk — potentially raising total drawdown, the very thing the 15% target exists to control.
3. **Two overlays fighting.** The commodity sleeve already vol-targets (and may carry its own HMM crash damper). Literally moving notional into a self-vol-targeting book could double-scale or conflict. Accounting must be at the **portfolio-return** level (blend the two sleeves' return streams), not by piping raw notional into a book that re-sizes it.
4. **Direction is not "buy commodities."** In a deflationary risk-off (2008), the trend book is often net **short** commodities. Routing equity capital in funds the *strategy's* return (direction-agnostic), which is fine — but the intuition "stocks down → buy commodities" is wrong and must not sneak into the design.
5. **Costs spike in stress.** The transfer creates extra turnover in *both* books timed to exactly when spreads/slippage are widest.

**How to test it (falsification-first):**
- **Mechanism:** dynamic commodity weight `w_comm_t = w_base + κ·(1 − e_t^eq)`, with `κ ∈ [0,1]` (κ=0 recovers the static book; κ=1 fully redeploys the sidelined slice). Signal `e_t^eq` is already computed and lookahead-free.
- **Baselines it must beat, out-of-sample after costs:** (a) equity-to-**cash** + **static** ~15% commodity sleeve (the naive combination), (b) equity-to-cash, no commodities, (c) constant-mix rebalanced. If it can't beat (a), the dynamic routing adds nothing.
- **Metrics:** total-portfolio Sharpe **and** MaxDD, skew, worst-month, plus the *stress-month* return specifically. The claim to falsify: routing-to-commodities improves **drawdown-adjusted** return vs routing-to-cash.
- **Robustness / event study:** isolate the handful of vol-spike episodes (2018, 2020, 2022) and check the **fast-reversal** case explicitly — does it get double-whipsawed?
- **Placebo/null:** replace the commodity sleeve with cash or a *random* diversifier of matched vol; the improvement must come from **real crisis alpha**, not the mechanical act of staying invested.

**Prior:** promising *because* the de-risk trigger and the crisis alpha share the same stress driver — but it must clear baseline (a) and survive the fast-reversal whipsaw, or it's just re-levering risk back in at the worst time. Test only *after* the real commodity book (steps 1–8) exists, since it depends on the true trend-sleeve stress return, not the ETF proxy.

## 14. Out of scope for now

- No intraday / HFT; this is a monthly (trend possibly weekly) systematic-macro sleeve.
- No single-market directional discretionary betting.
- No physical delivery — cash-settled or rolled well before first-notice/delivery.
- Shared capital allocation with the other sleeves is decided *after* the real-book gate, not assumed.

## 15. Key reading

- Erb & Harvey (2006), *The Strategic and Tactical Value of Commodity Futures* — roll yield is the return driver.
- Gorton & Rouwenhorst (2006), *Facts and Fantasies about Commodity Futures* — the foundational diversification/inflation case.
- Koijen, Moskowitz, Pedersen & Vrugt (2018), *Carry* — carry across asset classes incl. commodities.
- Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum* — the trend / crisis-alpha backbone.
- Fuertes, Miffre & Rallis (2010), *Tactical Allocation… Combining Momentum and Term-Structure Signals* — the carry+trend combo we build.
- Gorton, Hayashi & Rouwenhorst (2013), *The Fundamentals of Commodity Futures Returns* — inventory/theory-of-storage (why carry pays).
- Tang & Xiong (2012), *Index Investment and the Financialization of Commodities* — the correlation-rose caveat.
