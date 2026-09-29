# Contributing

## Setup

```bash
git clone https://github.com/heykav/microprice-rust.git
cd microprice-rust
# Linux only: the CLI's plotting dependency needs fontconfig headers.
sudo apt-get install libfontconfig1-dev
```

## Checks (the same ones CI runs)

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

The Python bindings are a separate Cargo workspace; build them with
`maturin` as described in [`docs/python-bindings.md`](docs/python-bindings.md)
and run `docs/python_bindings_smoke_test.py`.

## Ground rules

- **No unsupported numbers.** Accuracy, error or speed figures go in the
  README or docs only if they come from a reproducible command that is
  written next to them, on real data or on data explicitly labelled
  synthetic. Report negative results as negative.
- **Fixtures are format tests.** Hand-written vendor-format snippets exist to
  test parsing. Label them as such; never present them as results.
- **Do not commit vendor data** (Binance, LOBSTER or others). Downloads go in
  the git-ignored `data/` directory.
- **Pre-registration.** Changes to the evaluation protocol in
  `docs/real-data-evaluation.md` should be made before looking at the results
  they would affect, and noted in the changelog.
- Library code: no `unsafe` (`#![forbid(unsafe_code)]`), typed errors, no
  `unwrap()`/`expect()` outside tests.
- Do not weaken CI (lints, warnings-as-errors, test scope) to make a change
  pass.
- Add an entry to `CHANGELOG.md` under Unreleased. Releases and tags are
  the maintainer's decision.

## Pull requests

Branch from `main`, keep a PR to one coherent change, and make sure the checks
above pass locally.
