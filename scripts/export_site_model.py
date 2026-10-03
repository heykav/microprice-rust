#!/usr/bin/env python3
"""Regenerate the `const MODEL = {...}` block embedded in site/index.html.

Trains the demo model with the project's CLI on its deterministic SYNTHETIC
generator (seed 42; not market data), then reads every state's G* and
training-visit count back through `microprice predict`, one representative
book per state, and checks that the book encoded to the expected state id.
Bucket ranges are recomputed with the same formulas as
`StateSpaceConfig::decode` (imbalance `[b/N, (b+1)/N)`, spread bounds as
configured). Only the one `const MODEL = ...` line is replaced.

Usage (from the repo root; standard library only):

    python3 scripts/export_site_model.py

Environment: MICROPRICE_BIN (path to a built `microprice` binary; otherwise
`cargo build --release -p microprice-cli` is run), CARGO_TARGET_DIR.
`predict` prints G* with 6 decimals, which is the precision embedded.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site" / "index.html"
SEED = 42
NUM_EVENTS = 500_000
NUM_IMB = 20
BOUNDS = [1, 2, 4]
ALPHA = 0.5  # the CLI default, passed explicitly so the page's label is exact


def binary() -> str:
    if os.environ.get("MICROPRICE_BIN"):
        return os.environ["MICROPRICE_BIN"]
    subprocess.run(["cargo", "build", "--release", "-p", "microprice-cli"], cwd=ROOT, check=True)
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    return str(target / "release" / "microprice")


def run(*args: str) -> str:
    res = subprocess.run(args, check=True, capture_output=True, text=True)
    return res.stdout + res.stderr


def spread_ranges():
    """(lo, hi) per spread bucket, hi None for the unbounded tail."""
    out, lo = [], 1
    for b in BOUNDS:
        out.append((lo, b))
        lo = b + 1
    out.append((lo, None))
    return out


def main():
    bin_ = binary()
    ranges = spread_ranges()
    states = []
    with tempfile.TemporaryDirectory() as tmp:
        model = os.path.join(tmp, "site.bin")
        run(bin_, "train", "--output", model, "--seed", str(SEED),
                        "--num-events", str(NUM_EVENTS), "--num-imbalance-buckets", str(NUM_IMB),
                        "--spread-bucket-bounds", ",".join(map(str, BOUNDS)),
                        "--smoothing-alpha", str(ALPHA))
        observations = int(re.search(r"training_observations:\s+(\d+)",
                                     run(bin_, "inspect", "--model", model)).group(1))
        width = 1.0 / NUM_IMB
        for sb, (s_lo, s_hi) in enumerate(ranges):
            for ib in range(NUM_IMB):
                sid = sb * NUM_IMB + ib
                imb = (ib + 0.5) / NUM_IMB
                bid_q, ask_q = round(1000 * imb), round(1000 * (1 - imb))
                out = run(bin_, "predict", "--model", model,
                          "--bid-price-ticks", "10000", "--bid-qty", str(bid_q),
                          "--ask-price-ticks", str(10000 + s_lo), "--ask-qty", str(ask_q))
                got = int(re.search(r"state_id:\s+(\d+)", out).group(1))
                if got != sid:
                    sys.exit(f"representative book for state {sid} encoded to {got}")
                states.append({
                    "state_id": sid, "imbalance_bucket": ib, "spread_bucket": sb,
                    "imbalance_lo": ib * width, "imbalance_hi": (ib + 1) * width,
                    "spread_lo": s_lo, "spread_hi": s_hi,
                    "g_star": float(re.search(r"adjustment_ticks:\s+(\S+)", out).group(1)),
                    "visits": int(re.search(r"state_observations:\s+(\d+)", out).group(1)),
                })
    blob = {"num_imbalance_buckets": NUM_IMB, "num_spread_buckets": len(ranges),
            "spread_bucket_bounds_ticks": BOUNDS, "smoothing_alpha": ALPHA,
            "training_observations": observations, "states": states}
    html = SITE.read_text()
    new, n = re.subn(r"^const MODEL = .*;?$", lambda _: "const MODEL = " + json.dumps(blob),
                     html, count=1, flags=re.M)
    if n != 1:
        sys.exit("could not find the `const MODEL = ...` line in site/index.html")
    SITE.write_text(new)
    print(f"updated {SITE} ({len(states)} states, "
          f"{sum(s['visits'] > 0 for s in states)} visited in training)")


if __name__ == "__main__":
    main()
