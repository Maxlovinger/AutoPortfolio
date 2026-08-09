% The Two Strategies — Equations & a Worked Monthly Walkthrough
% autoPortfolio: technical reference
% 2026-08-08

# 0. Notation

| Symbol | Meaning |
|---|---|
| $r_i$ | daily (or monthly) return of asset $i$ |
| $w_i$ | portfolio weight on asset $i$ |
| $\hat\sigma_{t}$ | estimated (trailing, realized) volatility at time $t$, annualized |
| $e_t$ | **exposure** at $t$ — fraction of the book that is invested (rest is cash); $e_t\in[0,1]$ |
| $g$ | gross exposure of the currency book ($=1$) |
| $c$ | per-name cost (half-spread) in basis points (bps); 1 bp $=0.01\%$ |

Two layers everywhere: **weights** (how the invested money is split) and
**exposure** (how much is invested at all). They are computed separately.

\newpage

# 1. Equity strategy — the equations

## 1.1 Selection (quarterly)

Score every eligible point-in-time member by liquidity (dollar volume), then take
the top $N=30$ subject to a **sector cap** (at most $M=5$ per sector). Formally the
book is
$$
\mathcal{B} = \operatorname*{top\text{-}N}_{\text{sector cap }M}\big(\text{liquidity score}\big).
$$

## 1.2 Weights — equal weight

Each selected name gets the same weight:
$$
w_i = \frac{1}{N} = \frac{1}{30} \approx 3.33\%,\qquad \sum_{i\in\mathcal{B}} w_i = 1 .
$$

## 1.3 Exposure — the 15% volatility target

This is the **equity exposure equation.** Measure the book's trailing realized
volatility and scale exposure so risk stays near a constant target
$\sigma_\text{tgt}=15\%$:
$$
\boxed{\,e_t^{\text{eq}} \;=\; \min\!\left(1,\ \frac{\sigma_\text{tgt}}{\hat\sigma_{t-1}}\right)\,}
\qquad
\hat\sigma_{t-1} = \operatorname{std}\!\big(r_{t-21},\dots,r_{t-1}\big)\times\sqrt{252}.
$$

- $\hat\sigma$ uses only **past** returns (a 21-trading-day window), lagged one
  step, so there is no look-ahead.
- The $\min(1,\cdot)$ means we **never lever** — exposure only ever falls below
  100%; the shortfall $\,1-e_t\,$ sits in cash.
- Calm markets ($\hat\sigma<15\%$) $\Rightarrow e_t=1$ (fully invested). Turbulent
  markets ($\hat\sigma>15\%$) $\Rightarrow e_t<1$ (de-risked).

The final invested weight on name $i$ is $\,e_t \cdot w_i$.

## 1.4 The "water-filling" cap algorithm

*Where it is used:* enforcing a **per-name cap** $c_\text{max}$ when a weighting
scheme would otherwise concentrate too much in one name (e.g. the sentiment-tilt
experiments). The deployed **equal-weight** book never triggers it, because
$1/30=3.3\%$ is already below any sensible cap — but here is the exact rule,
because plain "clip then renormalize" both **violates** the cap and **flattens**
distinct weights.

Given raw weights $w$ (summing to 1) and cap $c_\text{max}$, repeat until no
weight exceeds the cap:

$$
\begin{aligned}
&\text{Let } O=\{i: w_i>c_\text{max}\}. \quad\text{If } O=\varnothing,\text{ stop.}\\
&\text{Excess: } E=\sum_{i\in O}(w_i-c_\text{max}).\\
&\text{Cap them: } w_i \leftarrow c_\text{max}\ \ \forall i\in O.\\
&\text{Redistribute proportionally to the uncapped names:}\\
&\qquad w_j \leftarrow w_j + E\cdot\frac{w_j}{\sum_{k\notin O} w_k}\quad \forall j\notin O.
\end{aligned}
$$

Each pass moves the "overflow" above the cap line down and pours it into the
names still below the line (hence *water-filling*), **preserving their relative
ordering**. It converges in a few passes and always yields $\sum_i w_i=1$ with
every $w_i\le c_\text{max}$ (feasible whenever $N\cdot c_\text{max}\ge 1$).

\newpage

# 2. Currency strategy — the equations

## 2.1 The signal — carry

For each currency, carry is its 3-month interest rate minus the US dollar's,
annualized in percent:
$$
\text{carry}_i \;=\; r_i^{3M} - r_\text{USD}^{3M}.
$$
Positive $\Rightarrow$ higher-yielding than the dollar (a *long* candidate);
negative $\Rightarrow$ lower-yielding (a *short* candidate).

## 2.2 Weights — rank, then long/short, dollar-neutral

Rank all currencies by carry. Go **long the top $n$** and **short the bottom
$n$**, equal weight within each sleeve:
$$
\boxed{\,
w_i=\begin{cases}
+\dfrac{g/2}{n} & i\in\text{top-}n\ \text{(highest carry)}\\[2mm]
-\dfrac{g/2}{n} & i\in\text{bottom-}n\ \text{(lowest carry)}\\[1mm]
0 & \text{otherwise}
\end{cases}\,}
$$
with gross $g=1$. Then
$$
\sum_i w_i = 0 \ \ (\text{dollar-neutral}),\qquad \sum_i |w_i| = g = 1 .
$$
With $n=3$: each long weight $=+\tfrac{0.5}{3}=+0.1667$, each short $=-0.1667$.

## 2.3 Return accounting — carry-inclusive

Holding a foreign currency (funded in dollars) earns the **spot move plus the
interest differential** over the holding period $\Delta t=\tfrac{1}{12}$ (monthly):
$$
\boxed{\,R_i \;=\; \underbrace{\frac{S_{i,t}}{S_{i,t-1}}-1}_{\text{spot return}}
\;+\; \underbrace{\frac{\text{carry}_i}{100}\cdot \Delta t}_{\text{interest earned}}\,}
$$
The book return, net of trading cost, is
$$
R^{\text{book}}_t=\sum_i w_i R_i,\qquad
\text{cost}_t=\sum_i \big|w_i-w_i^{\text{prev}}\big|\cdot\frac{c_i}{10^4},
$$
where $c_i$ is that currency's half-spread in bps (majors $\approx5$, EM $\approx30$).

## 2.4 Exposure — the HMM crash overlay (optional)

This is the **currency exposure equation.** A 2-state (calm / stress) Gaussian
Hidden Markov Model is fit on crash features (cross-currency realized vol;
safe-haven vs high-beta return spread). Let $P(\text{stress}\mid \text{obs}\le t)$
be its **filtered** probability (forward algorithm, frozen parameters — uses only
past data). Then
$$
\boxed{\,e_t^{\text{fx}} \;=\; \operatorname{clip}\!\big(1-P(\text{stress}\mid \text{obs}\le t-1),\ 0,\ 1\big)\,}
$$
so exposure falls toward 0 as the stress probability rises. The final book return
with the overlay is
$$
R^{\text{net}}_t \;=\; e_t^{\text{fx}}\cdot R^{\text{book}}_t \;-\; \big|e_t^{\text{fx}}-e_{t-1}^{\text{fx}}\big|\cdot\frac{c}{10^4}.
$$

\newpage

# 3. Worked example — a full monthly reallocation

Using the **current** data (latest month).

## 3.1 Currency book — step by step

**Step 1 — signal.** From FRED, compute each currency's carry vs USD. The extremes
of the ranking:

| Currency | carry vs USD | Role |
|---|---|---|
| ZAR (S. Africa) | $+3.34\%$ | long |
| MXN (Mexico) | $+2.99\%$ | long |
| HUF (Hungary) | $+2.21\%$ | long |
| SEK (Sweden) | $-1.82\%$ | short |
| JPY (Japan) | $-2.45\%$ | short |
| CHF (Switzerland) | $-3.81\%$ | short |

**Step 2 — weights.** $n=3$, so each long $=+\tfrac{0.5}{3}=+0.1667$, each short
$=-0.1667$ (sum $=0$, gross $=1$).

**Step 3 — realize the month.** For each name, $R_i=\text{spot ret}+\text{carry}/1200$:

| Currency | $w_i$ | spot ret | + carry accrual | $=R_i$ | $w_iR_i$ |
|---|---|---|---|---|---|
| CHF | $-0.1667$ | $-0.24\%$ | $-0.318\%$ | $-0.558\%$ | $+0.093\%$ |
| JPY | $-0.1667$ | $+1.55\%$ | $-0.204\%$ | $+1.346\%$ | $-0.224\%$ |
| SEK | $-0.1667$ | $+0.66\%$ | $-0.151\%$ | $+0.509\%$ | $-0.085\%$ |
| HUF | $+0.1667$ | $+0.13\%$ | $+0.185\%$ | $+0.315\%$ | $+0.053\%$ |
| MXN | $+0.1667$ | $+1.34\%$ | $+0.249\%$ | $+1.589\%$ | $+0.265\%$ |
| ZAR | $+0.1667$ | $+2.31\%$ | $+0.278\%$ | $+2.588\%$ | $+0.431\%$ |

**Step 4 — book return.** Sum the last column:
$$
R^{\text{book}} = 0.093-0.224-0.085+0.053+0.265+0.431 = \mathbf{+0.53\%}\ \text{(this month)}.
$$

*What this shows:* the short-JPY leg **lost** ($-0.224\%$) because the yen rallied
(a safe-haven move) — that is exactly the carry risk. But the EM longs (MXN, ZAR)
more than covered it. In a true risk-off month, the yen would spike **and** the EM
longs would crash together — that is the negative-skew "carry crash," and it is
why the exposure overlay ($e^{\text{fx}}_t$) exists.

**Step 5 — costs & exposure.** Turnover is tiny (the ranking rarely changes), so
cost is a fraction of a bp. If the HMM reads "calm," $e^{\text{fx}}=1$ and the net
return is $\approx+0.53\%$; if it read stress at $P=0.4$, exposure would be
$e=1-0.4=0.6$ and we would realize $0.6\times0.53\%=0.32\%$.

## 3.2 Equity book — step by step

**Step 1 — holdings (quarterly).** Select the 30 most-liquid point-in-time names,
capped at 5 per sector, equal weight $w_i=1/30=3.33\%$ each.

**Step 2 — exposure (checked monthly/weekly).** Measure trailing 21-day realized
vol. Currently
$$
\hat\sigma_{t-1}=16.9\%\ \Rightarrow\ e^{\text{eq}}_t=\min\!\left(1,\ \frac{0.15}{0.169}\right)=\mathbf{0.89}.
$$
So the book is **89% invested, 11% cash.** Each name's live weight is
$0.89\times3.33\%=2.97\%$.

**Step 3 — if markets turn.** Say vol jumps to $25\%$ next month:
$$
e^{\text{eq}}=\min\!\left(1,\frac{0.15}{0.25}\right)=0.60
\ \Rightarrow\ \text{cut to 60\% invested, 40\% cash},
$$
each name to $0.60\times3.33\%=2.0\%$. The trades to get there are the only
turnover; the holdings themselves don't change until the next quarterly rebalance.

# 4. One-glance summary of the equations

**Equity weight** — equal weight on the sector-capped selection:
$$w_i = 1/N .$$

**Equity exposure** — 15% volatility target:
$$e^{\text{eq}}_t=\min\big(1,\ \sigma_\text{tgt}/\hat\sigma_{t-1}\big),\qquad \sigma_\text{tgt}=0.15 .$$

**Water-fill cap** — cap the over-limit names at $c_\text{max}$, pour the excess
$E$ into the uncapped names in proportion to their weight ($w_j \propto w_j$),
repeat until all $w_i\le c_\text{max}$.

**Currency weight** — long/short the top/bottom-$n$ by carry, dollar-neutral:
$$w_i=\pm\frac{g/2}{n}\ \text{(else 0)},\qquad \sum_i w_i=0,\ \ \sum_i|w_i|=g .$$

**Currency return** — carry-inclusive:
$$R_i=\Big(\tfrac{S_{i,t}}{S_{i,t-1}}-1\Big)+\frac{\text{carry}_i}{1200},\qquad
R^{\text{book}}=\sum_i w_iR_i-\text{cost} .$$

**Currency exposure** — HMM crash overlay:
$$e^{\text{fx}}_t=\operatorname{clip}\big(1-P(\text{stress}),\,0,\,1\big) .$$
