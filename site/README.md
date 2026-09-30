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

1. A model was trained on the project's **synthetic** generator (not market data) with the project's CLI:
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

**What this means in practice:** the numbers you see come from an
actual run of the trainer, but on synthetic data, so they say nothing about
real markets. Real-data validation is pre-registered
([`docs/real-data-evaluation.md`](../docs/real-data-evaluation.md)) but has
not been run. The formula applied to your inputs is the same one the Rust
crate uses — but editing the code here doesn't change what the actual Rust
crate does, and this page can't retrain or evaluate on new data. For
that, use `microprice train`/`evaluate` (see the repository root
`README.md`).

## SEO metadata on this page

`index.html`'s `<head>` carries a canonical link, Open Graph + Twitter
Card tags (image: `og-image.png`, an actual `microprice visualize` render of the synthetic-trained model,
not a stock graphic), a `SoftwareSourceCode` JSON-LD block, and a CSP
matching this project's other public surfaces. `robots.txt` and
`sitemap.xml` sit alongside `index.html` for the same reason.

## Regenerating the embedded data

```bash
python3 scripts/export_site_model.py   # standard library only; builds the CLI unless MICROPRICE_BIN is set
```

It retrains the model with the command above (synthetic generator, seed 42),
reads each of the 80 states' `G*` and visit count back through
`microprice predict` (checking every representative book encodes to the
intended state), and rewrites only the `const MODEL = ...` line of
`index.html`. `G*` is embedded at the 6 decimals `predict` prints. The
embedded data was last regenerated this way after the smoothed-`G1`
denominator fix (`CHANGELOG.md`); `og-image.png` was rendered before that
fix and was not regenerated.
