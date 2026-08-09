% The Foreign-Equity Strategy — A Plain-English Design
% autoPortfolio: International Stocks (design / proposal, not yet built)
% 2026-08-09

# What this document is

This is the **blueprint** for a third strategy: a **foreign (international) stock**
portfolio to sit alongside the two you already run — US equities and FX carry. It
is a *design*, not a deployed system. The goal here is to explain, in plain
English, **exactly how it would work**, and — just as importantly — to be honest
about **whether it would actually help**, because for one large chunk of the
foreign universe the honest answer is probably *"barely."*

Everything reuses the machinery you already trust. The interesting, new questions
are only two: **the currency** and **the diversification**. Those get most of the
space.

\newpage

# Part 1 — The core: same recipe as US equities

The foreign book is deliberately the **same structure** as your US equity book,
because that structure already survived heavy testing (simple beat clever, every
time). One sentence:

> Hold an **equal-weighted basket of the most-liquid foreign stocks**, refreshed
> **quarterly**, diversified across sectors **and regions**, with the same **15%
> volatility-target** safety overlay — and, on top, a **currency decision**.

The five pieces you already know, applied abroad:

**1. Which stocks? The most *liquid* foreign names.**
Same as home: rank eligible foreign stocks by dollar trading volume, take the top
*N*. Liquid names keep trading costs realistic — and abroad this matters *more*,
because foreign markets are generally less liquid and more expensive to trade than
the US.

**2. How much of each? Equal weight (1/N).**
Same reason it won at home: 1/N is famously hard to beat out-of-sample. No change.

**3. Diversified across sectors *and regions* (a cap on each).**
At home we cap **≤5 per sector**. Abroad we add a second cap: **≤ some max per
country/region**, so the book can't quietly become an all-Japan or all-Europe
bet. This is the one genuinely new selection rule.

**4. The 15% volatility-target overlay.**
Identical: measure the book's recent volatility, move to cash when it spikes, stay
near a steady 15%/yr. Never levers, only de-risks.

**5. Point-in-time + null tests.**
Same discipline: at every past date use the names that were *actually* liquid
*then* (no hindsight), test every design choice against a random-signal "null,"
validate on held-out periods. Nothing is trusted until it beats coin-flips.

If that were the whole story, this page would be the whole document. It isn't,
because of the two foreign-specific problems below.

\newpage

# Part 2 — New problem #1: the currency (hedge or not)

A foreign stock is really **two bets bundled together**:

> foreign-stock return (in USD) ≈ the local stock move **+** the currency's move vs the dollar

Buy a Japanese stock: even if it rises 10% in yen, a 10% fall in the yen wipes the
gain when you convert back to dollars. So owning foreign stocks *unhedged* means
you're **also** running a pile of currency bets — whether you meant to or not.

That's a problem for *you specifically*, because **you already run a currency
book** (the carry sleeve). Stacking unmanaged FX exposure on top of a deliberately
managed FX sleeve means you could be **taking the same yen or peso risk twice**, in
sizes you never chose.

**The fix — hedge the currency.** Hedging means removing the FX portion (sell the
foreign currency forward, same size as the position), leaving a **clean bet on the
foreign companies**. The two sleeves then stay in their lanes: the equity sleeve
bets on *stocks*, the carry sleeve bets on *currencies*, and neither accidentally
doubles the other.

| | **Unhedged** | **Hedged (recommended)** |
|---|---|---|
| You own | Stock **+** currency bet | Just the stock |
| Cost | Free | Small: the interest-rate gap between the two currencies |
| Overlaps carry book? | Yes — doubles FX risk | No — clean separation |
| Adds FX diversification? | Yes, but uncontrolled | No |

**Recommendation: hedge, at least for developed markets.** It keeps the risk
attributable to one place and makes the diversification question (next) *clean* —
we can ask "do foreign *companies* diversify US companies?" without the answer
being muddied by currency noise. One caveat: hedging **costs** roughly the carry
differential, and for EM that cost is large — so for an EM sub-book we'd test
hedged **and** unhedged and let the data decide.

\newpage

# Part 3 — New problem #2: does it even diversify? (the real question)

Running a third strategy is only worth it if it **doesn't move with the first
two**. This is where foreign equities split into two very different animals.

## Developed international (Europe, Japan, UK, Canada, Australia…)

The uncomfortable truth: **developed international stocks are ~0.8 correlated with
US stocks.** They fall in the same crashes (2008, 2020, 2022) and rise in the same
booms. They are *almost the same bet* in a slightly different wrapper.

This is the **exact same situation as G10 carry**, which the Markowitz optimizer
gave **0% weight** — not because it was bad, but because it was too *similar* to
what you already held to earn a slice. Developed-international equity will very
likely hit the **same wall**: the optimizer looks at it, sees "US stocks with
extra steps," and allocates near zero.

So we should *expect* developed international to earn a **small-to-zero
allocation**, and we should be fine with that. Testing it is still worthwhile — but
we go in expecting the answer "it's redundant," not hoping otherwise.

## Emerging-market equity (the interesting one)

EM equity (China, India, Brazil, Taiwan, Korea…) is the **genuine diversifier
candidate** — it's only ~0.6–0.7 correlated with US stocks, driven by different
growth stories, commodity cycles, and local politics. This is the **direct
analogue of widening carry to EM**, which is precisely the move that *worked*
(carry Sharpe 0.31 → 0.51, and it earned a real 29% Markowitz slice).

But EM equity carries **its own crash tail** — and here's the subtle trap: it may
crash **at the same time and for the same reason** as your **EM carry** book. When
the Mexican peso devalues, Mexican stocks usually fall too. So EM equity and EM
carry could share a **hidden common tail** — the diversification looks great in
calm data and then *both* lose together in exactly the crisis you were hedging.

**This is the one thing we must measure before committing a dollar.**

## The gate it has to pass (before we build)

Same diversification threshold we applied to the carry sleeve: **an asset earns a
slice only if its risk-adjusted return clears the bar set by how correlated it is
to what you already own.** Concretely, before building we run:

1. **Correlation check** — foreign book vs US equity **and** vs the carry book,
   overall *and in each sleeve's worst months* (the tail correlation is what
   matters, not the average).
2. **Markowitz with three assets** — US equity, carry, foreign — and see what
   weight foreign actually earns. If it's ~0 (likely for developed), we learn that
   *cheaply, before writing the engine.*
3. **Shared-tail check for EM** — does EM equity crash *with* EM carry? If yes,
   the "diversification" is partly an illusion and we size it down hard.

\newpage

# Part 4 — What the finished strategy would look like

Putting it together, the deployed foreign book (if it passes the gate):

| Piece | Choice | Same as US? |
|---|---|---|
| **Universe** | Most-liquid foreign stocks, split developed vs EM | new: two regions |
| **Selection** | Top-N by dollar volume, point-in-time | same |
| **Weighting** | Equal weight (1/N) | same |
| **Diversification cap** | ≤ M per sector **and** ≤ K per country/region | new: region cap |
| **Rebalance** | Quarterly | same |
| **Risk overlay** | 15% volatility target (de-risk to cash) | same |
| **Currency** | **Hedged** (developed); test both for EM | **new** |
| **Validation** | Point-in-time, null tests, held-out periods | same |

## Honest expected verdict (before any data)

- **Developed international, hedged:** clean, safe, well-diversified *internally* —
  but ~0.8 correlated to US, so expect a **near-zero** portfolio slice. Probably
  **not worth deploying** on its own merits; it's "more of the same."
- **EM equity:** the real prize *if* it survives the shared-tail test with EM
  carry. If its crashes are independent enough, it earns a genuine slice like EM
  carry did. If they crash together, it's a smaller diversifier than it looks and
  gets sized down. **This is the piece worth building — pending the test.**
- **Overall:** the value of this strategy is **concentrated in EM equity**, and
  even there it's gated on one specific measurement. So the right first step is
  **not to build the engine** — it's to run the three-asset correlation/Markowitz
  check with a cheap proxy (e.g. regional index returns) and confirm foreign equity
  clears the bar. Build only what the diversification math says is worth building.

# One-page summary

| Question | Answer |
|---|---|
| **Core recipe?** | Same as US equity: liquid, equal-weight, sector-capped, quarterly, 15% vol-target, point-in-time |
| **What's new?** | (1) a **currency hedge** decision, (2) a **region cap**, (3) a developed-vs-EM split |
| **Hedge the currency?** | **Yes** for developed (keeps FX risk in the carry sleeve, not doubled); test both for EM |
| **Does developed intl diversify?** | Barely — ~0.8 correlated to US, likely ~0% optimal weight (like G10 carry) |
| **Does EM equity diversify?** | Potentially yes (~0.6 correlated) — the real prize — **but** may share a crash tail with EM carry |
| **What must we check first?** | 3-asset Markowitz + **tail**-correlation, especially EM-equity vs EM-carry, *before* building |
| **Expected bottom line** | Value is concentrated in **EM equity**, gated on the shared-tail test; developed intl is likely redundant |

*This is a design document. No foreign-equity engine has been built yet. The
recommended first action is the diversification gate (Part 3) using cheap index
proxies — build the full point-in-time engine only for the pieces that clear it.*
