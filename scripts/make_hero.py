#!/usr/bin/env python3
"""Generate the animated README visuals as plain SVG (standard library only).

Outputs, in docs/img/:
  hero-{dark,light}.svg          banner with the animated order-book schematic
  imbalance-{dark,light}.svg     larger explainer: schematic plus the calibration steps
  social-preview.svg             static 1280x640 card (PNG via scripts/make_hero.py --png)

Design rules (these SVGs are shown through <img> on GitHub, so no script runs):
  * CSS keyframes only, no SMIL, no per-character positioning, no textLength.
    All text is static font text; only whole groups fade or translate.
  * The base style is the finished state (bid-heavy book, marker off the mid).
    The animation starts and ends on that same state, and
    prefers-reduced-motion turns it off, so a still render is the final frame.
  * Everything drawn as a book or marker is an ILLUSTRATION of the concept and is
    labelled that way. It is not data and not model output. The marker is placed
    at the size-weighted mid purely for intuition.

Usage: python3 scripts/make_hero.py [--png /path/to/chrome]
"""
import argparse
import os
import subprocess
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "img")

MONO = "ui-monospace,'SFMono-Regular','DejaVu Sans Mono',Menlo,Consolas,monospace"
SERIF = "Georgia,'Times New Roman',serif"
SANS = "system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"

THEMES = {
    "dark": dict(bg="#090A09", panel="#111211", line="#2a2c29", grid="#1a1b1a", text="#F4F5F2",
                 sub="#c7c9c4", mute="#9a9c96", acc="#00FF66", acc2="#B6FF2E", bid="#00FF66",
                 ask="#B6FF2E", askop="0.55", warn_bg="#14180f", dots=("#ff5f57", "#febc2e", "#28c840")),
    "light": dict(bg="#F4F5F2", panel="#eceeea", line="#c4c7c0", grid="#e2e3df", text="#111211",
                  sub="#3d3f3b", mute="#54564f", acc="#00A84A", acc2="#5f9a00", bid="#00A84A",
                  ask="#5f9a00", askop="0.6", warn_bg="#e6eadd", dots=("#ff5f57", "#febc2e", "#28c840")),
}

# Illustrative states: bar heights (px) for 4 levels away from the mid, best level first.
# Imbalance I = bid/(bid+ask) at the best level; marker shift = (I - 0.5) * spread.
SPREAD = 80
STATES = {
    "a": dict(bid=[140, 100, 70, 45], ask=[46, 70, 95, 115], imb="0.75", tag="bid-heavy", shift=20,
              note="more size on the bid: the next move is more likely up"),
    "b": dict(bid=[90, 80, 70, 55], ask=[90, 80, 70, 55], imb="0.50", tag="balanced", shift=0,
              note="equal queues: no pressure, micro-price sits on the mid"),
    "c": dict(bid=[46, 70, 95, 115], ask=[140, 100, 70, 45], imb="0.25", tag="ask-heavy", shift=-20,
              note="more size on the ask: the next move is more likely down"),
}

CSS = """
.pa{opacity:1}.pb{opacity:0}.pc{opacity:0}.mk{transform:translateX(20px)}
@keyframes fa{0%,28%{opacity:1}34%,88%{opacity:0}94%,100%{opacity:1}}
@keyframes fb{0%,28%{opacity:0}34%,56%{opacity:1}62%,100%{opacity:0}}
@keyframes fc{0%,56%{opacity:0}62%,84%{opacity:1}90%,100%{opacity:0}}
@keyframes mv{0%,28%{transform:translateX(20px)}36%,56%{transform:translateX(0)}64%,84%{transform:translateX(-20px)}93%,100%{transform:translateX(20px)}}
.an .pa{animation:fa 12s ease-in-out infinite}
.an .pb{animation:fb 12s ease-in-out infinite}
.an .pc{animation:fc 12s ease-in-out infinite}
.an .mk{animation:mv 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.an .pa,.an .pb,.an .pc,.an .mk{animation:none}}
"""


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def t(x, y, s, size, fill, family=MONO, weight="normal", anchor="start", extra=""):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" {extra}>{esc(s)}</text>')


def window(c, x, y, w, h, title):
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{c["panel"]}" stroke="{c["line"]}"/>']
    for i, d in enumerate(c["dots"]):
        s.append(f'<circle cx="{x + 22 + i * 18}" cy="{y + 20}" r="5" fill="{d}"/>')
    s.append(t(x + w / 2, y + 25, title, 12, c["mute"], anchor="middle"))
    s.append(f'<line x1="{x}" y1="{y + 40}" x2="{x + w}" y2="{y + 40}" stroke="{c["line"]}"/>')
    return "".join(s)


def book(c, cx, base, bw=34, gap=16):
    """Order-book schematic centred on x=cx with its price axis at y=base. Returns SVG fragment."""
    s = [f'<line x1="{cx - 230}" y1="{base}" x2="{cx + 230}" y2="{base}" stroke="{c["line"]}" stroke-width="1.5"/>']
    for k in (50, 100, 150):
        s.append(f'<line x1="{cx - 230}" y1="{base - k}" x2="{cx + 230}" y2="{base - k}" stroke="{c["grid"]}"/>')
    for key, st in STATES.items():
        g = [f'<g class="p{key}">']
        for i in range(4):
            right = cx - SPREAD / 2 - i * (bw + gap)
            h = st["bid"][i]
            g.append(f'<rect x="{right - bw}" y="{base - h}" width="{bw}" height="{h}" rx="3" fill="{c["bid"]}" '
                     f'opacity="{1 - i * 0.18:.2f}"/>')
            left = cx + SPREAD / 2 + i * (bw + gap)
            h = st["ask"][i]
            g.append(f'<rect x="{left}" y="{base - h}" width="{bw}" height="{h}" rx="3" fill="{c["ask"]}" '
                     f'opacity="{1 - i * 0.18:.2f}"/>')
        g.append("</g>")
        s.append("".join(g))
    s.append(f'<line x1="{cx}" y1="{base - 170}" x2="{cx}" y2="{base + 8}" stroke="{c["text"]}" stroke-width="1.5" '
             f'stroke-dasharray="4 4"/>')
    s.append(t(cx, base - 178, "mid", 12, c["mute"], anchor="middle"))
    s.append(f'<g class="mk"><line x1="{cx}" y1="{base - 150}" x2="{cx}" y2="{base}" stroke="{c["acc"]}" '
             f'stroke-width="2.5"/><path d="M{cx} {base + 6} l-7 12 h14 z" fill="{c["acc"]}"/>'
             + t(cx, base + 40, "micro-price", 12, c["acc"], anchor="middle", weight="bold") + "</g>")
    s.append(t(cx - 230, base + 18, "bids", 12, c["bid"]))
    s.append(t(cx + 230, base + 18, "asks", 12, c["ask"], anchor="end"))
    return "".join(s)


def readouts(c, x, y, size=14):
    s = []
    for key, st in STATES.items():
        s.append(f'<g class="p{key}">' + t(x, y, f'imbalance {st["imb"]}  ({st["tag"]})', size, c["text"], weight="bold")
                 + "</g>")
    return "".join(s)


def notes(c, x, y, size=12):
    return "".join(f'<g class="p{k}">' + t(x, y, st["note"], size, c["sub"]) + "</g>" for k, st in STATES.items())


def head(c, w, h, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-labelledby="t d"><title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>'
            f'<style>{CSS}</style><rect width="{w}" height="{h}" fill="{c["bg"]}"/>')


def status_box(c, x, y, w):
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="76" rx="8" fill="{c["warn_bg"]}" stroke="{c["line"]}"/>',
         f'<rect x="{x}" y="{y}" width="5" height="76" rx="2" fill="{c["acc2"]}"/>',
         t(x + 22, y + 26, "STATUS", 12, c["acc2"], weight="bold"),
         t(x + 22, y + 48, "One real day (Binance BTCUSDT, 2024-01-15) and synthetic data:", 14, c["text"], SANS),
         t(x + 22, y + 66, "the micro-price does not beat the naive mid on either.", 14, c["text"], SANS)]
    return "".join(s)


def hero(c):
    w, h = 1280, 420
    s = [head(c, w, h, "MicroPrice-Rust",
              "Banner for MicroPrice-Rust, queue-imbalance micro-price estimation for limit order books in Rust. "
              "An animated order-book schematic shows the micro-price moving off the mid as queue imbalance changes. "
              "The schematic is illustrative, not data. Status: research code. One real-data result (Binance BTCUSDT, one day) and synthetic data: "
              "the micro-price does not beat the naive mid on either.")]
    s.append('<g class="an">')
    s.append(f'<rect x="0.75" y="0.75" width="{w - 1.5}" height="{h - 1.5}" rx="14" fill="none" stroke="{c["line"]}" stroke-width="1.5"/>')
    s.append(t(64, 74, "RESEARCH CODE  /  NEGATIVE RESULT ON REAL DATA", 13, c["acc"], weight="bold"))
    s.append(t(64, 144, "MicroPrice", 64, c["text"], SERIF, "bold") .replace("</text>", "") +
             f'<tspan fill="{c["acc"]}">-Rust</tspan></text>')
    s.append(t(66, 190, "Queue-imbalance micro-price estimation", 22, c["sub"], SANS))
    s.append(t(66, 220, "for limit order books, in Rust.", 22, c["sub"], SANS))
    s.append(t(66, 262, "mid + G*(imbalance, spread)", 17, c["acc2"]))
    s.append(status_box(c, 64, 296, 560))
    s.append(t(66, 398, "github.com/heykav/microprice-rust", 13, c["mute"]))
    s.append(window(c, 676, 40, 556, 340, "orderbook.schematic   (illustrative, not data)"))
    s.append(readouts(c, 704, 110))
    s.append(book(c, 954, 292))
    s.append(notes(c, 704, 366))
    s.append("</g></svg>")
    return "".join(s)


def explainer(c):
    w, h = 1280, 560
    s = [head(c, w, h, "How a micro-price differs from the mid",
              "Illustrative schematic, not data. An animated order book where queue imbalance moves the micro-price "
              "off the mid, next to the four calibration steps the library follows: discretize quotes into "
              "imbalance and spread states, estimate a Markov chain, solve for the G star adjustment, output mid plus G star.")]
    s.append('<g class="an">')
    s.append(f'<rect x="0.75" y="0.75" width="{w - 1.5}" height="{h - 1.5}" rx="14" fill="none" stroke="{c["line"]}" stroke-width="1.5"/>')
    s.append(t(48, 56, "HOW IT WORKS  /  ILLUSTRATIVE, NOT DATA", 13, c["acc"], weight="bold"))
    s.append(t(48, 92, "Queue imbalance pushes the fair price off the mid", 28, c["text"], SERIF, "bold"))
    s.append(window(c, 48, 116, 640, 400, "orderbook.schematic"))
    s.append(readouts(c, 80, 190, 16))
    s.append(book(c, 368, 398, 40, 18))
    s.append(notes(c, 80, 496, 13))
    steps = [("1", "Discretize", "Each quote becomes a state:", "(imbalance bucket, spread bucket)."),
             ("2", "Estimate", "Count state transitions and mid", "moves to build a Markov chain."),
             ("3", "Solve", "Iterate the recursion to a fixed", "point: the adjustment G*(state)."),
             ("4", "Output", "micro-price = mid + G*(state)", "an estimate of the future mid.")]
    for i, (n, a, b1, b2) in enumerate(steps):
        y = 116 + i * 100
        s.append(f'<rect x="720" y="{y}" width="512" height="88" rx="10" fill="{c["panel"]}" stroke="{c["line"]}"/>')
        s.append(f'<circle cx="758" cy="{y + 44}" r="16" fill="none" stroke="{c["acc"]}" stroke-width="2"/>')
        s.append(t(758, y + 50, n, 16, c["acc"], weight="bold", anchor="middle"))
        s.append(t(790, y + 34, a, 18, c["text"], SANS, "bold"))
        s.append(t(790, y + 56, b1, 14, c["sub"], SANS))
        s.append(t(790, y + 74, b2, 14, c["sub"], SANS))
    s.append(t(720, 540, "Does not beat the naive mid on synthetic data or one real day of BTCUSDT.", 13, c["acc2"], SANS))
    s.append("</g></svg>")
    return "".join(s)


def social():
    c = THEMES["dark"]
    w, h = 1280, 640
    s = [head(c, w, h, "MicroPrice-Rust", "Social preview card for MicroPrice-Rust. Illustrative schematic, not data.")]
    s.append(t(80, 150, "RESEARCH CODE  /  NEGATIVE RESULT ON REAL DATA", 16, c["acc"], weight="bold"))
    s.append(t(80, 240, "MicroPrice", 72, c["text"], SERIF, "bold").replace("</text>", "") +
             f'<tspan fill="{c["acc"]}">-Rust</tspan></text>')
    s.append(t(82, 310, "Queue-imbalance micro-price", 30, c["sub"], SANS))
    s.append(t(82, 350, "estimation for limit order books.", 30, c["sub"], SANS))
    s.append(t(82, 400, "mid + G*(imbalance, spread)", 22, c["acc2"]))
    s.append(status_box(c, 80, 450, 600))
    s.append(t(82, 590, "github.com/heykav/microprice-rust", 17, c["mute"]))
    s.append(window(c, 716, 100, 500, 400, "orderbook.schematic  (illustrative)"))
    s.append(book(c, 966, 400, 30, 12))
    s.append(readouts(c, 744, 170, 15))
    s.append("</svg>")
    return "".join(s)  # no "an" wrapper class: static, finished state only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", help="Chrome binary; also writes docs/img/social-preview.png and site/og-image.png")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for name, c in THEMES.items():
        for stem, fn in (("hero", hero), ("imbalance", explainer)):
            with open(os.path.join(OUT, f"{stem}-{name}.svg"), "w") as f:
                f.write(fn(c) + "\n")
    with open(os.path.join(OUT, "social-preview.svg"), "w") as f:
        f.write(social() + "\n")
    if a.png:
        svg = os.path.abspath(os.path.join(OUT, "social-preview.svg"))
        out = os.path.abspath(os.path.join(OUT, "social-preview.png"))
        subprocess.run([a.png, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                        "--window-size=1280,640", f"--screenshot={out}", f"file://{svg}"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
