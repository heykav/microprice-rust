#!/usr/bin/env bash
# Fetch one day of Binance USD-M futures bookTicker data from the public
# data.binance.vision archive (no credentials) and run the real-data
# evaluation described in docs/real-data-evaluation.md.
#
#   TICK_SIZE=0.1 scripts/fetch_binance_bookticker.sh BTCUSDT 2024-01-15
#
# Environment:
#   TICK_SIZE       required. Price of one tick for the symbol (never guessed;
#                   check exchangeInfo PRICE_FILTER.tickSize).
#   MAX_EVENTS      keep only the first N events (chronological prefix; a
#                   memory guard). Default 5000000. Set to 0 for all rows.
#   SKIP_DOWNLOAD   if 1, use an existing data/<SYMBOL>-bookTicker-<DATE>.csv.
#   DRY_RUN         if 1, print what would run and exit.
#
# The downloaded data stays in the git-ignored data/ directory. Vendor data
# is not redistributed by this repository.
set -euo pipefail

usage() {
  sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-2}"
}

[[ $# -eq 2 ]] || usage
SYMBOL="$1"
DATE="$2"
[[ "$SYMBOL" =~ ^[A-Z0-9_]+$ ]] || { echo "error: SYMBOL must look like BTCUSDT" >&2; exit 2; }
[[ "$DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { echo "error: DATE must be YYYY-MM-DD" >&2; exit 2; }
: "${TICK_SIZE:?error: set TICK_SIZE (price of one tick for $SYMBOL); it is not guessed}"
MAX_EVENTS="${MAX_EVENTS:-5000000}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
DRY_RUN="${DRY_RUN:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${SYMBOL}-bookTicker-${DATE}"
BASE_URL="https://data.binance.vision/data/futures/um/daily/bookTicker/${SYMBOL}"
DATA_DIR="${ROOT}/data"
CSV="${DATA_DIR}/${NAME}.csv"
REPORT="${ROOT}/results/binance-bookticker-${SYMBOL}-${DATE}.md"

EVAL_CMD=(cargo run --release -p microprice-cli -- evaluate-csv
  --input "$CSV" --format binance-bookticker --tick-size "$TICK_SIZE"
  --skip-invalid-rows --report-md "$REPORT")
if [[ "$MAX_EVENTS" != "0" ]]; then
  EVAL_CMD+=(--max-events "$MAX_EVENTS")
fi

echo "url:     ${BASE_URL}/${NAME}.zip"
echo "csv:     ${CSV}"
echo "report:  ${REPORT}"
echo "command: ${EVAL_CMD[*]}"
[[ "$DRY_RUN" == "1" ]] && exit 0

for tool in cargo; do
  command -v "$tool" >/dev/null || { echo "error: $tool not found" >&2; exit 1; }
done

mkdir -p "$DATA_DIR" "${ROOT}/results"
if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  for tool in curl unzip; do
    command -v "$tool" >/dev/null || { echo "error: $tool not found" >&2; exit 1; }
  done
  ZIP="${DATA_DIR}/${NAME}.zip"
  curl -fL --retry 3 -o "$ZIP" "${BASE_URL}/${NAME}.zip"
  # The archive publishes a checksum file next to each zip. If it cannot be
  # fetched or parsed we say so rather than silently skipping verification.
  if curl -fsL -o "${ZIP}.CHECKSUM" "${BASE_URL}/${NAME}.zip.CHECKSUM"; then
    expected="$(awk '{print $1}' "${ZIP}.CHECKSUM" | head -n1)"
    if command -v sha256sum >/dev/null; then
      actual="$(sha256sum "$ZIP" | awk '{print $1}')"
    else
      actual="$(shasum -a 256 "$ZIP" | awk '{print $1}')"
    fi
    if [[ "$expected" != "$actual" ]]; then
      echo "error: checksum mismatch for ${ZIP} (expected ${expected}, got ${actual})" >&2
      exit 1
    fi
    echo "checksum ok: ${actual}"
  else
    echo "warning: could not fetch ${NAME}.zip.CHECKSUM; download NOT verified" >&2
  fi
  unzip -o -q "$ZIP" -d "$DATA_DIR"
fi
[[ -f "$CSV" ]] || { echo "error: expected ${CSV} after download/unzip" >&2; exit 1; }
echo "first lines of ${CSV} (confirm the header matches docs/real-data-evaluation.md):"
head -n 3 "$CSV"

cd "$ROOT"
"${EVAL_CMD[@]}"
echo "wrote ${REPORT}"
