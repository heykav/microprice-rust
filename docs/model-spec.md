# MicroPrice-Rust — Model Specification (v0)

This document fixes the vocabulary and precise definitions this project uses
before any of the estimation machinery is built. Everything here is either a
definition (no ambiguity possible) or an explicit modeling decision (stated as
one, with its rationale) — anything genuinely undecided is pushed to
[Open Questions](#open-questions) rather than quietly assumed.

This is v0: it covers what Phase 1 (primitive types) needs to be unambiguous.
It will grow as later phases (state encoding, transition estimation, the
solver) require their own precise definitions — those sections are stubbed
below and will be filled in when those phases start, not written speculatively
now.

## Price and quantity representation

Prices are represented as **integer ticks**, not floating point. For an
instrument with `tick_size = $0.01`, a price of `$187.32` is represented as
`PriceTicks(18732)`. Rationale: exact equality/ordering comparisons, no
binary/decimal rounding drift accumulating over millions of updates, and
spread arithmetic (`ask_ticks - bid_ticks`) is exact integer subtraction
instead of a floating-point subtraction that needs a tolerance.

Quantities are non-negative integers: `Quantity(u64)`. A quantity is a count
of shares/contracts/lots at a price level; there is no such thing as a
negative resting quantity, so the type itself rules that out rather than
validating it at runtime.

## Mid-price

Given a valid top-of-book with best bid price `Pb` and best ask price `Pa`:

```text
M = (Pb + Pa) / 2
```

Mid-price is defined here as a value in **half-tick units** internally when
`Pb + Pa` is odd (i.e. we do not silently truncate); see
[`PriceTicks` arithmetic](#open-questions) for how this is exposed.

## Spread

```text
S = Pa - Pb
```

`S` is measured in ticks. A **valid** (non-crossed, non-locked) book has
`S > 0`. `S = 0` is a **locked** market (`Pb == Pa`); `S < 0` is a **crossed**
market (`Pb > Pa`). Crossed and locked books are real (if rare) states a raw
feed can be in — this project treats accepting or rejecting them as a
*configurable validation policy* on `TopOfBook` construction, not a hidden
assumption. See [`BookValidationPolicy`](#topofbook-validation).

## Queue imbalance

```text
I = Qb / (Qb + Qa)
```

where `Qb` is quantity resting at the best bid and `Qa` is quantity resting at
the best ask. By construction `0 <= I <= 1`. `I -> 1` means the bid-side queue
dominates (more resting size wants to buy at the touch than sell); `I -> 0`
means the ask-side dominates.

**Degenerate cases**, decided explicitly rather than left to whatever the
floating-point division happens to produce:

| Case | `I` |
|---|---|
| `Qb = 0`, `Qa > 0` | `0` (exact — `0 / Qa`) |
| `Qa = 0`, `Qb > 0` | `1` (exact — `Qb / Qb`) |
| `Qb = 0` **and** `Qa = 0` | **undefined — construction fails** with `MicroPriceError::EmptyBook` |

The zero/zero case is not a design choice this project papers over with an
arbitrary constant like `0.5`: a book with nothing resting on either side of
the touch isn't a state this model has any empirical transition data for
anyway, and inventing a number for it would be exactly the kind of "arbitrary
heuristic" the project's engineering philosophy explicitly rules out. Callers
that need to handle empty books do so explicitly at the `TopOfBook`
construction boundary, not by receiving a silently-guessed imbalance.

## Weighted mid-price (baseline, not the model)

```text
WeightedMid = Pa * (Qb / (Qb + Qa)) + Pb * (Qa / (Qb + Qa))
            = M + (S / 2) * (2I - 1)
```

This is the naive size-weighted mid, implemented as a baseline for sanity
checking and benchmarking — **not** the micro-price. It requires no
calibration and captures none of the empirically-estimated transition
dynamics that make micro-price different from "a fancier mid-price formula."

## Micro-price (target quantity — definition fixed, estimation deferred)

```text
MicroPrice = MidPrice + G(state)
```

where `G(state)` is an empirically-calibrated expected-future-mid-price
adjustment conditional on the current discretized order-book state — *not* a
closed-form function of `I` and `S` the way `WeightedMid` is. Precisely what
"state" means (which buckets, what else besides imbalance/spread it
conditions on) and precisely how `G` is estimated (the transition-matrix
solve) are **Phase 2 and Phase 5 concerns respectively** and are deliberately
not fixed in this document — see [Open Questions](#open-questions).

## State (V1, as of Phase 3 / Prompt 2)

V1 state is exactly two dimensions: an **imbalance bucket** and a **spread
bucket**, packed into one contiguous `StateId`:

```text
state_id = spread_bucket * num_imbalance_buckets + imbalance_bucket
```

**Imbalance bucketing** is uniform: `num_buckets` equal-width, half-open
intervals over `[0.0, 1.0]`, i.e. `[k/N, (k+1)/N)` for bucket `k`, with the
final bucket closed on the right so `I == 1.0` lands in bucket `N-1` rather
than the out-of-range index `N`. Lookup is `floor(I * N)` clamped to
`N - 1` — one multiply, one floor, one clamp, no search.

**Spread bucketing** is explicit, not uniform: a strictly increasing list
of inclusive tick upper bounds for every bucket except the last, which is
always unbounded above. `SpreadBucketing::new(vec![1, 2, 4])` produces the
four buckets `{1}`, `{2}`, `{3,4}`, `{5, 6, ...}` — matching this
document's earlier `[[1],[2],[3,4],[5,inf]]` example exactly. This shape is
deliberate: it makes construction reject any bound configuration that
could leave a "gap" a valid spread might fall through, which in turn makes
state *encoding* infallible for its spread lookup on any spread `>= 1`
tick — only a locked/crossed book (spread `< 1` tick, only reachable via a
non-default `BookValidationPolicy`) can fail encoding, and it fails with a
named error (`SpreadOutOfRange`), not a panic or a guessed bucket.

Both bucketings' `MicroPriceError::InvalidBucketConfig` construction
failures happen at configuration time — `ImbalanceBucketing::new(0)` and a
non-increasing or sub-1-tick `SpreadBucketing` bound list are both rejected
before any book is ever encoded, per the Phase 3 requirement that a bad
configuration must fail at construction, not confusingly during
prediction.

This resolves Open Question #2 below for V1: no order-flow or volatility-
regime dimension yet, uniform (not quantile/adaptive) imbalance buckets.
Both remain legitimate later extensions, not implemented here.

## Transition / price-changing transition (V1, as of Phase 5)

V1 uses **event-to-event sampling**: consecutive `BookEvent`s from the same
symbol, in `sequence` order, form one observed transition `state_i ->
state_j`. This is the simplest of the sampling modes the project brief
lists (fixed-time-interval and N-event-horizon sampling are legitimate,
undecided-for-V1 alternatives — see Open Questions) and is chosen because
it needs no additional configuration to be well-defined.

**Price-changing event definition: mid-price crossing.** A transition is
classified by comparing `mid_price_ticks()` at `t` and `t+1`:

```text
delta = mid_price_ticks(t+1) - mid_price_ticks(t)

delta > 0  ->  UP
delta < 0  ->  DOWN
delta == 0 ->  UNCHANGED
```

This is a specific, deliberate choice among the brief's listed candidates
(mid-price crossing vs. tick-normalized movement vs. best-quote depletion)
— **not** an accident of which one was easiest to code. Best-quote
depletion in particular is a real, arguably more information-rich
alternative (section 25 of the project brief) that this project is
choosing *not* to use for V1 classification, because it conflates two
distinct things (a queue emptying vs. the price actually moving) that
mid-price crossing keeps cleanly separate: a book can deplete one side to
zero while `mid_price_ticks()` stays exactly where it was (the other side
hasn't moved), which V1 correctly calls `UNCHANGED`.

`delta` itself (not just its sign) is retained per-transition and is what
Phase 7's `G1` estimation actually averages — see the
[Micro-price estimation](#micro-price-estimation-v1-as-of-phase-7-8) section
below. This is a deliberate generalization beyond "assume every
price-changing move is exactly one tick": the synthetic generator (Phase
4) only ever produces single-tick moves, but the estimator does not
hard-code that assumption, so it degrades correctly against data with
multi-tick jumps.

## Micro-price estimation (V1, as of Phase 7/8)

For each state `i`, transition counting (Phase 5) accumulates, over every
observed `state_i -> state_j` transition:

- `visits[i]`: total number of times state `i` was the *starting* state of
  a transition.
- `count[i][j]`: how many of those transitions landed in state `j`.
- `delta_sum[i]`: the sum of every observed `delta` (signed tick change,
  see above) over transitions starting at `i`.

From these:

```text
Q[i][j] = count[i][j] / visits[i]              (for j reached with delta == 0)
G1[i]   = delta_sum[i] / visits[i]
```

`Q` is therefore **sub-stochastic** by construction (`sum_j Q[i][j] <= 1`):
its rows only cover the *non-price-changing* destinations, and the missing
probability mass is exactly `P(price changed | i)`. `G1[i]` is the
one-step expected mid-price change from state `i` — an average over *all*
transitions from `i`, not just the price-changing ones (a transition that
doesn't change price contributes exactly `0` to the sum, which is the
mathematically correct way to fold "how often does the price even move
from here" into a single one-step expectation, rather than needing a
separate up/down-probability formula).

The full micro-price adjustment solves the recursive relationship the
project brief specifies:

```text
G*[i] = G1[i] + sum_j Q[i][j] * G*[j]
```

i.e. `G* = G1 + Q @ G*`, equivalently `G* = (I - Q)^{-1} G1` — **computed
via fixed-point iteration** (`G*_0 = G1`, `G*_{k+1} = G1 + Q @ G*_k`, until
`max|G*_{k+1} - G*_k| < tolerance` or a max-iteration budget is hit),
never by explicitly inverting `(I - Q)`, per the project brief's explicit
instruction. Non-convergence within the iteration budget is a named,
returned error (`SolverError::DidNotConverge`), not a silently-truncated
result.

`MicroPrice = MidPrice + G*[StateSpaceConfig::encode(book)]`.

## Smoothing (V1, as of Phase 7)

Raw counts can leave `visits[i] == 0` for states never observed during
calibration (`count[i][j]` and `delta_sum[i]` are then both `0/0`).
V1 supports **additive (Laplace-style) smoothing**: a configurable
`alpha >= 0.0` added to every `count[i][j]` before normalizing, and to a
per-state pseudo-observation before computing `G1`. `alpha == 0.0` (no
smoothing) is valid and means a zero-observation state is flagged
(`InsufficientObservations`) rather than silently producing a `G1` of
exactly `0.0` that looks like "no adjustment," which would be a
misleading conflation of "never observed" with "observed to have no
effect." State-merging and minimum-observation-threshold smoothing
(alternatives the project brief also lists) are not implemented in V1.

## `TopOfBook` validation

Because "is this book valid" is itself a modeling decision (a locked market
is unusual but not impossible; whether to reject it depends on what the
caller is doing with the data), `TopOfBook` construction takes an explicit
`BookValidationPolicy`:

- `RejectCrossedAndLocked` — require `Pa > Pb` strictly (the default).
- `RejectCrossedAllowLocked` — require `Pa >= Pb`.
- `AllowAll` — no price-ordering check at all (for research on raw/malformed
  feeds where the crossed/locked events are themselves the object of study).

There is no implicit default silently chosen deep in a function signature;
every call site names the policy it wants.

## Open Questions

Genuinely undecided items, to be resolved when the phase that needs them
starts — recorded here so "we haven't decided yet" is visible instead of
implicit:

1. **Half-tick mid-prices.** When `Pb + Pa` is odd, the true mid lands on a
   half-tick. Does `PriceTicks` gain a fixed-point/half-tick variant, does
   mid-price become its own type distinct from `PriceTicks`, or do we accept
   a documented rounding rule? Not decided — Phase 2 (core types beyond the
   Phase 1 primitives) needs to resolve this before `microprice()` can return
   a mid-price-derived value with a precise type.
2. ~~**State dimensionality beyond L1.**~~ **Resolved for V1** (Phase 3 /
   Prompt 2 — see the [State](#state-v1-as-of-phase-3--prompt-2) section
   above): imbalance + spread only, uniform imbalance buckets, explicit
   spread bucket bounds. Order flow, volatility regime, multi-level depth,
   and quantile/adaptive imbalance bucketing (which would need real
   calibration data to fit against, which this project doesn't have loaded
   yet) all remain legitimate later extensions, not V1 scope.
3. ~~**Price-changing event definition.**~~ **Resolved for V1** (Phase 5/6 —
   see [Transition / price-changing transition](#transition--price-changing-transition-v1-as-of-phase-5)):
   mid-price crossing, with the signed tick delta retained (not just
   up/down/unchanged), event-to-event sampling (not fixed-time-interval or
   N-event-horizon — those remain legitimate, unimplemented alternative
   *models*, not a superset this one subsumes).
4. ~~**Solver method.**~~ **Resolved for V1** (Phase 8 — see
   [Micro-price estimation](#micro-price-estimation-v1-as-of-phase-7-8)):
   fixed-point iteration on `G* = G1 + Q @ G*`, explicitly never a matrix
   inverse. Convergence tolerance and iteration budget are configurable;
   non-convergence is a returned error, not a truncated result.
5. ~~**Smoothing method.**~~ **Resolved for V1** (Phase 7 — see
   [Smoothing](#smoothing-v1-as-of-phase-7)): additive/Laplace smoothing
   only, configurable `alpha`. State-merging and minimum-observation-count
   thresholds are documented-but-unimplemented alternatives, not silently
   folded into the additive-smoothing option.
6. **Event sampling mode beyond event-to-event.** Fixed-wall-clock-interval
   and N-event-horizon sampling (both named in the project brief) are not
   implemented — `docs/model-spec.md`'s own rule against blurring sampling
   modes together means adding either later is a new, separate estimator
   path, not a generalization of the event-to-event one.
