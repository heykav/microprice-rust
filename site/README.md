# Interactive demo (`index.html`)

Deployed at **https://heykav.github.io/microprice-rust/** by
[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) on every
push to this directory. Also a single, self-contained HTML file — open it
directly in a browser (`open site/index.html`), no server, no build step.

This directory is deliberately **not** `docs/`: that directory already
holds this project's real markdown documentation (`model-spec.md`,
`benchmarking.md`, ...) and must never be handed to Jekyll/Pages
processing as if it were a site.

## What it actually is

A **static snapshot of one trained model**, not a live connection to the
Rust binary:

1. A real model was trained with the real pipeline:
   ```bash
   microprice train --num-events 500000 --num-imbalance-buckets 20 \
       --spread-bucket-bounds "1,2,4" --seed 42
   ```
2. Its full calibrated `g_star`/`visits` grid (all 80 states, decoded
   imbalance/spread ranges included) was exported to JSON and embedded
   directly in `index.html` — there is no fetch, no backend, nothing to
   deploy beyond the static file itself.
3. The page reimplements the *same* state-encoding formula
   `microprice-core::StateSpaceConfig::encode` and `MicroPriceModel::predict`
   use (imbalance bucket via `floor(I * N)` clamped, spread bucket via the
   configured bounds, `state_id = spread_bucket * num_imbalance_buckets +
   imbalance_bucket`) in plain JavaScript, then looks the resulting state
   up in the embedded grid — verified by hand against the Rust source, not
   assumed to match.

**What this means in practice:** the numbers you see are real (from an
actual trained run), and the formula applied to your inputs is the real
one — but editing the code here doesn't change what the actual Rust
crate does, and this page can't retrain or evaluate on new data. For
that, use `microprice train`/`evaluate` (see the repository root
`README.md`).

## SEO metadata on this page

`index.html`'s `<head>` carries a canonical link, Open Graph + Twitter
Card tags (image: `og-image.png`, a real `microprice visualize` render,
not a stock graphic), a `SoftwareSourceCode` JSON-LD block, and a CSP
matching this project's other public surfaces. `robots.txt` and
`sitemap.xml` sit alongside `index.html` for the same reason.

## Regenerating the embedded data

There's no permanent export tool in this repo for this (it was a
one-off `cargo run --example` script, not committed, to avoid carrying
demo-specific plumbing in the library crates). To refresh `index.html`
against a different trained model: load the model with
`microprice_calibration::MicroPriceModel::load`, walk every `StateId` via
`model.state_space()?.decode(...)` alongside `model.g_star()`/`model.visits()`,
serialize the result, and replace the `const MODEL = {...}` block in
`index.html`.
