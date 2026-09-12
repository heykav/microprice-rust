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

## State (stub — filled in during Phase 3)

A `StateId` is an opaque, contiguous, deterministic integer index over some
discretization of order-book state. Phase 1 only introduces the *type*
(`StateId(u32)`) as a primitive; the mapping from a `TopOfBook` to a
`StateId` (bucket counts, bucketing method, dimensionality) is Phase 3's
concern and will get its own precise specification here when that phase
starts, not a preliminary guess now.

## Transition / price-changing transition (stub — filled in during Phase 5)

Deliberately not defined yet. The master project brief lists several
candidate definitions of a "price-changing" event (mid-price crossing,
tick-normalized movement, best-quote depletion) and is explicit that they are
*different models* and must not be blurred together. Fixing one now, before
Phase 5 needs it, would be exactly the kind of premature, unreviewed
mathematical commitment this project is trying to avoid.

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
2. **State dimensionality beyond L1.** Section 5/23 of the project brief
   allows for spread + imbalance only (V1) versus richer state (order flow,
   volatility regime, multi-level depth) later. V1's exact bucket scheme
   (uniform vs. quantile, bucket counts) is a Phase 3 decision, informed by
   real calibration data this project doesn't have loaded yet.
3. **Price-changing event definition.** As noted above — mid-price cross vs.
   tick-normalized movement vs. best-quote depletion. Phase 5/6.
4. **Solver method for `G* = (I - Q)^{-1} G1`.** Direct linear solve vs.
   iterative/sparse, and the exact numerical-stability guardrails (singular
   system detection, regularization) — Phase 5, and explicitly not to be
   decided by "whichever is easiest to write" but by benchmarking against the
   actual state-space sizes Phase 3 produces.
5. **Smoothing method for sparse/unobserved transitions** (Laplace smoothing
   vs. state merging vs. minimum-observation thresholds) — Phase 4/5, and
   configurable per the project brief rather than a single hardcoded choice.
