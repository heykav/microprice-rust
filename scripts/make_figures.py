#!/usr/bin/env python3
"""Regenerate the README figures from the project's own CLI output.

Every number plotted here is produced by running the `microprice` binary
(`train`, `predict`, `evaluate`) on the project's deterministic SYNTHETIC
generator (seed 42). Nothing is real market data and nothing is hard-coded
except the documented values used as a consistency check (see EXPECTED).

Usage (from the repo root):

    python3 -m venv /tmp/mpvenv && /tmp/mpvenv/bin/pip install matplotlib numpy
    /tmp/mpvenv/bin/python scripts/make_figures.py

Environment: CARGO_TARGET_DIR (as for cargo), MICROPRICE_BIN (skip the build).
Writes docs/img/fig-*-{dark,light}.png and docs/img/figures-data.json.
Output is deterministic for fixed tool versions (no timestamps in PNGs).
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"
SEED = 42
NUM_IMB = 10
SPREAD_BOUNDS = "1,2,4"
# Spread bucket -> a representative spread in ticks (bounds 1,2,4 are
# inclusive upper bounds; the last bucket is unbounded above).
SPREAD_REP = [1, 2, 3, 5]
SPREAD_LABEL = ["<= 1", "2", "3-4", ">= 5"]
# Values documented in the README; the script fails if a fresh run disagrees.
EXPECTED = {"microprice": 0.1094, "mid": 0.0988, "symmetrized": 0.1031}

THEMES = {
    "light": dict(bg="#ffffff", fg="#1f2328", muted="#59636e", grid="#d1d9e0",
                  c1="#0b6bcb", c2="#c2410c", c3="#0f766e", c4="#6e7781",
                  miss="#e6eaee", hatch="#9aa4ae"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", muted="#9198a1", grid="#30363d",
                 c1="#58a6ff", c2="#f0883e", c3="#3fb9a8", c4="#8b949e",
                 miss="#1b222b", hatch="#5b6570"),
}


def binary() -> str:
    if os.environ.get("MICROPRICE_BIN"):
        return os.environ["MICROPRICE_BIN"]
    subprocess.run(["cargo", "build", "--release", "-p", "microprice-cli"],
                   cwd=ROOT, check=True)
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    return str(target / "release" / "microprice")


def run(*args: str) -> str:
    res = subprocess.run(args, check=True, capture_output=True, text=True)
    return res.stdout + res.stderr


def surface(bin_: str, tmp: str, symmetrize: bool):
    """G* (ticks) and visit counts per (spread bucket, imbalance bucket),
    read back through `microprice predict`'s reported adjustment_ticks."""
    model = os.path.join(tmp, "sym.bin" if symmetrize else "raw.bin")
    cmd = [bin_, "train", "--output", model, "--seed", str(SEED),
           "--num-events", "500000", "--num-imbalance-buckets", str(NUM_IMB),
           "--spread-bucket-bounds", SPREAD_BOUNDS]
    if symmetrize:
        cmd.append("--symmetrize")
    run(*cmd)
    g = np.zeros((len(SPREAD_REP), NUM_IMB))
    visits = np.zeros_like(g)
    for si, spread in enumerate(SPREAD_REP):
        for ii in range(NUM_IMB):
            imb = (ii + 0.5) / NUM_IMB
            bid, ask = round(1000 * imb), round(1000 * (1 - imb))
            out = run(bin_, "predict", "--model", model,
                      "--bid-price-ticks", "10000", "--bid-qty", str(bid),
                      "--ask-price-ticks", str(10000 + spread), "--ask-qty", str(ask))
            g[si, ii] = float(re.search(r"adjustment_ticks:\s+(\S+)", out).group(1))
            visits[si, ii] = float(re.search(r"state_observations:\s+(\d+)", out).group(1))
    return g, visits


def evaluate(bin_: str, symmetrize: bool) -> dict:
    cmd = [bin_, "evaluate", "--seed", str(SEED), "--num-events", "300000",
           "--num-imbalance-buckets", str(NUM_IMB),
           "--spread-bucket-bounds", SPREAD_BOUNDS]
    if symmetrize:
        cmd.append("--symmetrize")
    out = run(*cmd)
    mae = re.search(r"MAE \(ticks\)\s+microprice=([\d.]+)\s+mid=([\d.]+)\s+weighted_mid=([\d.]+)", out)
    diag = re.search(r"fixed-point residual ([\d.e+-]+) ticks; .*?max \|E\[P'-P\|state\]\| = ([\d.e+-]+) ticks.*?"
                     r"visit-weighted mean ([\d.e+-]+) ticks", out)
    ne = re.search(r"n_evaluated / n_skipped:\s+(\d+) / (\d+)", out)
    held = re.search(r"(\d+) held-out test events", out)
    return dict(microprice=float(mae.group(1)), mid=float(mae.group(2)),
                weighted_mid=float(mae.group(3)), residual=float(diag.group(1)),
                drift_max=float(diag.group(2)), drift_mean=float(diag.group(3)),
                n_evaluated=int(ne.group(1)), n_held_out=int(held.group(1)))


def style(theme: str):
    t = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "text.color": t["fg"], "axes.labelcolor": t["fg"], "axes.edgecolor": t["grid"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False, "hatch.linewidth": 0.8,
    })
    return t


def save(fig, name: str, theme: str):
    fig.savefig(OUT / f"{name}-{theme}.png", dpi=160, metadata={"Software": None})
    plt.close(fig)


def stamp(fig, t, text):
    fig.text(0.012, 0.012, text, fontsize=8.5, color=t["muted"], ha="left", va="bottom")


def fig_heatmap(theme, gs, vs):
    t = style(theme)
    cmap = LinearSegmentedColormap.from_list("div", [t["c1"], t["bg"], t["c2"]])
    vmax = max(np.nanmax(np.abs(np.where(v > 0, g, np.nan))) for g, v in zip(gs, vs))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    titles = ["default calibration", "--symmetrize"]
    for ax, g, v, title in zip(axes, gs, vs, titles):
        for si in range(len(SPREAD_REP)):
            for ii in range(NUM_IMB):
                visited = v[si, ii] > 0
                ax.add_patch(plt.Rectangle((ii, si), 1, 1, lw=0.6, ec=t["bg"],
                             fc=cmap(norm(g[si, ii])) if visited else t["miss"],
                             hatch=None if visited else "///",
                             **({} if visited else {})))
                if not visited:
                    ax.patches[-1].set_edgecolor(t["hatch"])
                else:
                    c = cmap(norm(g[si, ii]))
                    lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
                    ax.text(ii + .5, si + .5, f"{g[si, ii]:+.3f}", ha="center", va="center",
                            fontsize=8, color="#111" if lum > 0.5 else "#fff")
        ax.set_xlim(0, NUM_IMB); ax.set_ylim(0, len(SPREAD_REP))
        ax.set_xticks(np.arange(NUM_IMB) + .5, [f"{i}" for i in range(NUM_IMB)])
        ax.set_xlabel("imbalance bucket  (0 = ask-heavy, 9 = bid-heavy)")
        ax.set_title(title, fontsize=11, color=t["fg"], loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(np.arange(len(SPREAD_REP)) + .5, SPREAD_LABEL)
    axes[0].set_ylabel("spread bucket (ticks)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=fig.add_axes([0.925, 0.2, 0.012, 0.6]))
    cb.set_label("G* (ticks), added to mid", color=t["fg"], fontsize=9.5)
    cb.outline.set_edgecolor(t["grid"]); cb.ax.tick_params(colors=t["muted"])
    fig.suptitle("Calibrated G* over (imbalance x spread): SYNTHETIC data, seed 42, 500,000 events",
                 x=0.012, ha="left", fontsize=12.5, color=t["fg"], fontweight="bold")
    n_unvisited = int((vs[0] == 0).sum())
    stamp(fig, t, f"Hatched = state never visited in training ({n_unvisited} of {vs[0].size}); "
                  "G* there is smoothing prior only and is not shown.\n"
                  "The generator has no imbalance-to-direction signal: the pattern above is an artifact "
                  "of this synthetic data, not evidence about markets.")
    fig.subplots_adjust(left=0.07, right=0.9, top=0.86, bottom=0.2, wspace=0.05)
    save(fig, "fig-gstar-heatmap", theme)


def fig_mae(theme, raw, sym):
    t = style(theme)
    rows = [("naive mid (baseline)", raw["mid"], t["c4"]),
            ("micro-price", raw["microprice"], t["c1"]),
            ("micro-price, --symmetrize", sym["microprice"], t["c3"]),
            ("size-weighted mid", raw["weighted_mid"], t["c2"])]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    for yi, r in zip(y, rows):
        ax.text(r[1] + 0.004, yi, f"{r[1]:.4f}", va="center", fontsize=10.5, color=t["fg"])
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 0.36)
    ax.set_xlabel("mean absolute error (ticks); lower is better")
    ax.xaxis.grid(True, color=t["grid"], lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    fig.suptitle("Held-out MAE at horizon 1: SYNTHETIC data, the micro-price does not beat naive mid",
                 x=0.012, ha="left", fontsize=12, color=t["fg"], fontweight="bold")
    stamp(fig, t, f"Output of `microprice evaluate`: seed 42, 300,000 events, 70/30 chronological split; "
                  f"{raw['n_held_out']:,} held-out events ({raw['n_evaluated']:,} evaluated).")
    fig.subplots_adjust(left=0.24, right=0.97, top=0.85, bottom=0.2)
    save(fig, "fig-mae-comparison", theme)


def fig_drift(theme, raw, sym):
    t = style(theme)
    labels = ["default calibration", "--symmetrize"]
    series = [("fixed-point residual", "residual", t["c4"]),
              ("visit-weighted mean |drift|", "drift_mean", t["c1"]),
              ("max |drift| over states", "drift_max", t["c2"])]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(2); w = 0.26
    for k, (name, key, col) in enumerate(series):
        vals = [raw[key], sym[key]]
        bars = ax.bar(x + (k - 1) * w, vals, w * 0.92, color=col, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.25, f"{v:.1e}", ha="center",
                    fontsize=8.5, color=t["fg"])
    ax.set_yscale("log"); ax.set_ylim(1e-12, 3e-2)
    ax.set_xticks(x, labels); ax.set_ylabel("ticks (log scale)")
    ax.yaxis.grid(True, color=t["grid"], lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    leg = ax.legend(frameon=False, loc="upper right", fontsize=9)
    for tx in leg.get_texts():
        tx.set_color(t["fg"])
    fig.suptitle("Solver residual vs one-step micro-price drift: SYNTHETIC data",
                 x=0.012, ha="left", fontsize=12, color=t["fg"], fontweight="bold")
    stamp(fig, t, "The residual is ~0 by construction; the drift is not, so the recursion is NOT a "
                  "martingale by construction\n(see docs/model-spec.md). Source: the diagnostic printed "
                  "by `microprice evaluate` (seed 42, 300,000 events).")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.2)
    save(fig, "fig-martingale-drift", theme)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bin_ = binary()
    with tempfile.TemporaryDirectory() as tmp:
        gs, vs = zip(surface(bin_, tmp, False), surface(bin_, tmp, True))
    raw, sym = evaluate(bin_, False), evaluate(bin_, True)
    for key, got in (("microprice", raw["microprice"]), ("mid", raw["mid"]),
                     ("symmetrized", sym["microprice"])):
        if abs(got - EXPECTED[key]) > 5e-5:
            sys.exit(f"documented {key} MAE {EXPECTED[key]} != regenerated {got}; "
                     "update the README or investigate")
    for theme in THEMES:
        fig_heatmap(theme, gs, vs)
        fig_mae(theme, raw, sym)
        fig_drift(theme, raw, sym)
    data = dict(note="SYNTHETIC data from the project's deterministic generator; not real market data",
                seed=SEED, evaluate_default=raw, evaluate_symmetrize=sym,
                g_star_default=gs[0].tolist(), visits_default=vs[0].tolist(),
                g_star_symmetrize=gs[1].tolist(), visits_symmetrize=vs[1].tolist(),
                spread_bucket_representative_ticks=SPREAD_REP)
    (OUT / "figures-data.json").write_text(json.dumps(data, indent=1) + "\n")
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
