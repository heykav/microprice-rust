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
#   ALLOW_UNVERIFIED  if 1, continue when the published .CHECKSUM cannot be
#                   fetched (default: refuse). The zip structure is still
#                   checked; authenticity is not.
#   DRY_RUN         if 1, print what would run and exit.
#
# The downloaded data stays in the git-ignored data/ directory. Vendor data
# is not redistributed by this repository.
set -euo pipefail

usage() {
  sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
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
# BASE_URL_OVERRIDE exists only to test the failure paths against a local
# stub server; it also permits plain http for that purpose.
if [[ -n "${BASE_URL_OVERRIDE:-}" ]]; then
  BASE_URL="${BASE_URL_OVERRIDE%/}"
  CURL_PROTO="=https,http"
else
  BASE_URL="https://data.binance.vision/data/futures/um/daily/bookTicker/${SYMBOL}"
  CURL_PROTO="=https"
fi
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

# Everything is downloaded and checked in a scratch directory next to the
# final location (same filesystem, so the final mv is atomic). Existing data
# is only replaced after every check has passed; any failure leaves it alone.
TMP="$(mktemp -d "${DATA_DIR}/.fetch.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

fail() { echo "error: $*" >&2; exit 1; }

# Prints the sha256 of a file, using whichever tool exists.
sha256_of() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

# fetch URL DEST: fail on HTTP errors, follow redirects, retry transient
# failures, and refuse an empty result.
fetch() {
  curl --fail --location --silent --show-error \
    --proto "$CURL_PROTO" --proto-redir "$CURL_PROTO" \
    --retry 3 --retry-delay 2 --connect-timeout 20 --max-time 900 \
    --output "$2" "$1" || return 1
  [[ -s "$2" ]] || { echo "error: empty response from $1" >&2; return 1; }
}

if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  for tool in curl unzip od; do
    command -v "$tool" >/dev/null || fail "$tool not found"
  done
  command -v sha256sum >/dev/null || command -v shasum >/dev/null \
    || fail "neither sha256sum nor shasum found; cannot verify the download"

  ZIP_TMP="${TMP}/${NAME}.zip"
  SUM_TMP="${TMP}/${NAME}.zip.CHECKSUM"

  fetch "${BASE_URL}/${NAME}.zip" "$ZIP_TMP" || fail "download of ${NAME}.zip failed"

  # Reject anything that is not a zip (HTML error pages, redirects to a
  # login/landing page, truncated bodies) before looking at it further.
  [[ "$(head -c 4 "$ZIP_TMP" | od -An -tx1 | tr -d ' \n')" == "504b0304" ]] \
    || fail "${NAME}.zip is not a zip file (missing PK header); first bytes: $(head -c 80 "$ZIP_TMP" | tr -c '[:print:]' '?')"
  unzip -tq "$ZIP_TMP" >/dev/null || fail "${NAME}.zip failed the zip integrity test"
  [[ "$(unzip -Z1 "$ZIP_TMP")" == "${NAME}.csv" ]] \
    || fail "${NAME}.zip must contain exactly ${NAME}.csv; found: $(unzip -Z1 "$ZIP_TMP" | tr '\n' ' ')"

  # The archive publishes a checksum file next to each zip. Without a
  # verified checksum we stop, unless the caller explicitly accepts that.
  if fetch "${BASE_URL}/${NAME}.zip.CHECKSUM" "$SUM_TMP"; then
    expected="$(awk 'NR==1 {print tolower($1)}' "$SUM_TMP")"
    listed="$(awk 'NR==1 {sub(/^\*/, "", $2); print $2}' "$SUM_TMP")"
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] \
      || fail "${NAME}.zip.CHECKSUM does not start with a sha256 digest; got: $(head -c 80 "$SUM_TMP" | tr -c '[:print:]' '?')"
    [[ -z "$listed" || "$listed" == "${NAME}.zip" ]] \
      || fail "${NAME}.zip.CHECKSUM names '${listed}', not '${NAME}.zip'"
    actual="$(sha256_of "$ZIP_TMP")"
    [[ "$expected" == "$actual" ]] \
      || fail "checksum mismatch for ${NAME}.zip (expected ${expected}, got ${actual})"
    echo "checksum ok: ${actual}"
  elif [[ "${ALLOW_UNVERIFIED:-0}" == "1" ]]; then
    echo "WARNING: could not fetch ${NAME}.zip.CHECKSUM; the download is NOT verified against a published checksum." >&2
    echo "         Proceeding only because ALLOW_UNVERIFIED=1. Zip structure was checked; content authenticity was not." >&2
    echo "         sha256 of what was downloaded: $(sha256_of "$ZIP_TMP")" >&2
  else
    echo "error: could not fetch ${BASE_URL}/${NAME}.zip.CHECKSUM." >&2
    echo "       Unverified: the integrity and origin of ${NAME}.zip (only its zip structure was checked)." >&2
    echo "       Refusing to continue. Set ALLOW_UNVERIFIED=1 to accept this explicitly." >&2
    exit 1
  fi

  unzip -q "$ZIP_TMP" -d "${TMP}/x"
  [[ -s "${TMP}/x/${NAME}.csv" ]] || fail "${NAME}.csv is empty after extraction"
  # A CSV starting with '<' is an HTML page that happened to be zipped.
  [[ "$(head -c 1 "${TMP}/x/${NAME}.csv")" != "<" ]] || fail "${NAME}.csv looks like HTML, not CSV"

  # All checks passed: move into place (atomic on the same filesystem).
  mv -f "$ZIP_TMP" "${DATA_DIR}/${NAME}.zip"
  mv -f "${TMP}/x/${NAME}.csv" "$CSV"
fi
[[ -s "$CSV" ]] || { echo "error: expected non-empty ${CSV} after download/unzip" >&2; exit 1; }
echo "first lines of ${CSV} (confirm the header matches docs/real-data-evaluation.md):"
head -n 3 "$CSV"

cd "$ROOT"
"${EVAL_CMD[@]}"
echo "wrote ${REPORT}"
