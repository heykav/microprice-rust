#!/usr/bin/env python3
"""Record N minutes of live Binance USD-M futures best bid/ask and evaluate it.

    TICK_SIZE=0.1 scripts/record_binance_live.py BTCUSDT 10

Reads the public `<symbol>@bookTicker` WebSocket stream (no key, no account)
and writes a CSV with the same columns as the data.binance.vision bookTicker
daily files, so the `binance-bookticker` preset reads it unchanged. Then it
runs `microprice evaluate-csv` with the settings of
docs/real-data-evaluation.md and writes results/live-<SYMBOL>-<UTC start>.md.

Needs the `websocket-client` package (pip install websocket-client).
The recording lands in the git-ignored data/ directory; do not redistribute
it (see the terms note in docs/real-data-evaluation.md).

Live stream fields (event type `bookTicker`): u update id, E event time (ms),
T transaction time (ms), s symbol, b/B best bid price/qty, a/A best ask
price/qty.
"""
import datetime
import json
import os
import subprocess
import sys
import time

try:
    import websocket
except ImportError:
    sys.exit("error: pip install websocket-client")

HEADER = "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time"


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    symbol, minutes = sys.argv[1].upper(), float(sys.argv[2])
    tick = os.environ.get("TICK_SIZE")
    if not tick:
        sys.exit("error: set TICK_SIZE (not guessed; see exchangeInfo PRICE_FILTER.tickSize)")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    csv_path = os.path.join(root, "data", f"live-{symbol}-{stamp}.csv")
    report = os.path.join(root, "results", f"live-{symbol}-{stamp}.md")

    url = f"wss://fstream.binance.com/ws/{symbol.lower()}@bookTicker"
    ws = websocket.create_connection(url, timeout=30)
    deadline = time.time() + minutes * 60
    rows = 0
    last_u = -1
    with open(csv_path, "w") as f:
        f.write(HEADER + "\n")
        while time.time() < deadline:
            m = json.loads(ws.recv())
            if m.get("e") != "bookTicker":
                continue
            if m["u"] <= last_u:  # keep the stream strictly ordered
                continue
            last_u = m["u"]
            f.write(f'{m["u"]},{m["b"]},{m["B"]},{m["a"]},{m["A"]},{m["T"]},{m["E"]}\n')
            rows += 1
    ws.close()
    print(f"recorded {rows} quotes to {csv_path}", flush=True)
    cmd = ["cargo", "run", "--release", "-p", "microprice-cli", "--", "evaluate-csv",
           "--input", csv_path, "--format", "binance-bookticker", "--tick-size", tick,
           "--skip-invalid-rows", "--report-md", report]
    print("command:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=root, check=True)
    print("wrote", report)


if __name__ == "__main__":
    main()
