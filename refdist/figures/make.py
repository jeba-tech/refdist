"""
Paper figures, drawn from the same tables the registered analysis uses.

  fig1_h1          FPR of every detector instance on every stratum vs. the
                   independent perplexity of that stratum (H1)
  fig2_h2          Fast-DetectGPT FPR, stratum x reference model (H2)
  fig3_h5          detector score change vs. perplexity change, one panel per
                   transform, with the common slope (H5)
  fig4_forecast    pre-registered forecast vs. actual FPR on S10-S12

Static, print-oriented. Colour palette validated (all-pairs, light surface):
blue/orange/aqua. Aqua is below 3:1 contrast, so every series also carries a
distinct marker shape and a direct label; colour never carries identity alone.

Usage:  python -m refdist.figures.make [--out data/results/figures]
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from refdist import config, paths  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "legend.frameon": False, "lines.linewidth": 2,
})


def _short(s: str) -> str:
    return config.STRATA[s]["name"].replace("StackExchange", "SE").replace("Gutenberg ", "Gut. ")


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def fig1_h1(cells: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, (fam, label) in enumerate((("zero_shot", "Zero-shot (scorer LM)"), ("trained", "Trained classifier"))):
        c = cells[cells["family"] == fam]
        ax.scatter(c["log_ppl_indep"], 100 * c["fpr"], s=22, marker=MARKERS[i], color=SERIES[i],
                   edgecolor=SURF, linewidth=0.8, label=label, zorder=3)
    b = np.polyfit(cells["log_ppl_indep"], 100 * cells["fpr"], 1)
    xs = np.linspace(cells["log_ppl_indep"].min(), cells["log_ppl_indep"].max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=INK2, linewidth=1.2, linestyle="--", label="Least-squares fit")
    per = cells.groupby("stratum")["log_ppl_indep"].mean().sort_values()
    span = float(per.max() - per.min()) or 1.0
    rows = range(6)  # stagger close labels into as many rows as the crowding needs
    last_x = {r: -1e9 for r in rows}
    for s, x in per.items():
        row = next((r for r in rows if (x - last_x[r]) / span > 0.03), rows[-1])
        last_x[row] = x
        ax.annotate(s, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2 + 8 * row),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xlabel("Independent log-perplexity of stratum (SmolLM2-360M; lower = more predictable)")
    ax.set_ylabel("False-positive rate (%)")
    ax.legend(loc="upper right", fontsize=7.5)
    _save(fig, out, "fig1_h1_unification")


def fig2_h2(cells: pd.DataFrame, out: Path, name: str = "fig2_h2_reference_shift",
            note: str = "") -> None:
    c = cells[cells["detector"] == "D1"]
    order = c.groupby("stratum")["log_ppl_indep"].mean().sort_values().index
    m = c.pivot_table(index="stratum", columns="reference", values="fpr").loc[order] * 100
    fig, ax = plt.subplots(figsize=(2.6 + 0.6 * m.shape[1], 0.32 * len(m) + 1.4))
    vmax = max(float(np.nanmax(m.to_numpy())), 1e-9)
    cmap = matplotlib.colors.ListedColormap(SEQ)
    ax.imshow(m.to_numpy(), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for (i, j), v in np.ndenumerate(m.to_numpy()):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if v / vmax > 0.55 else INK)
    corpus = {r: textwrap.fill(config.REFERENCE_MODELS[r]["corpus"], 10, break_long_words=False)
              for r in m.columns}
    ax.set_xticks(range(m.shape[1]), [f"{r}\n{corpus[r]}" for r in m.columns], fontsize=7.5)
    ax.set_yticks(range(m.shape[0]), [f"{s}  {_short(s)}" for s in m.index])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    fig.suptitle("Fast-DetectGPT false-positive rate (%),\nby reference model" + note, x=0.02, ha="left",
                 fontsize=9, color=INK)
    _save(fig, out, name)


def fig3_h5(deltas: pd.DataFrame, out: Path) -> None:
    keys = list(config.TRANSFORMS)
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.9), sharex=True, sharey=True)
    common = np.polyfit(deltas["dlogppl"], deltas["dz"], 1)
    xs = np.linspace(deltas["dlogppl"].quantile(0.005), deltas["dlogppl"].quantile(0.995), 20)
    for ax, k in zip(axes.flat, keys):
        g = deltas[deltas["transform"] == k]
        ax.scatter(g["dlogppl"], g["dz"], s=4, color=SERIES[0], alpha=0.25, linewidth=0, zorder=2)
        ax.plot(xs, np.polyval(common, xs), color=INK2, linewidth=1.2, linestyle="--", zorder=3)
        if len(g) > 2:
            ax.plot(xs, np.polyval(np.polyfit(g["dlogppl"], g["dz"], 1), xs), color=SERIES[0], zorder=4)
        ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
        ax.set_title(f"{k}: {config.TRANSFORMS[k]['name'].replace('_', ' ')}", loc="left", color=INK)
    for ax in axes[1]:
        ax.set_xlabel("Δ log-perplexity")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ detector score (SD)")
    fig.text(0.99, 0.005, "solid: this transform's slope   dashed: common slope", ha="right",
             fontsize=7, color=MUTED)
    _save(fig, out, "fig3_h5_transforms")


def fig4_forecast(cmp: pd.DataFrame, out: Path) -> None:
    # Legend sits outside the plot (identity = marker shape + colour + name), so no
    # in-plot labels compete with points piled up at forecast 0.
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    hi = 100 * max(cmp["fpr"].max(), cmp["fpr_pred"].max()) * 1.08 + 1
    ax.plot([0, hi], [0, hi], color=AXIS, linewidth=1, linestyle="--", zorder=1)
    for i, (s, g) in enumerate(cmp.groupby("stratum")):
        ax.scatter(100 * g["fpr_pred"], 100 * g["fpr"], s=26, marker=MARKERS[i % 3], color=SERIES[i % 3],
                   edgecolor=SURF, linewidth=0.8, zorder=3, label=f"{s} {_short(s)}", clip_on=False)
    ax.set(xlim=(-0.03 * hi, hi), ylim=(-0.03 * hi, hi), xlabel="Forecast FPR (%) — committed before scoring",
           ylabel="Observed FPR (%)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _save(fig, out, "fig4_forecast")


def make_all(out: Path) -> list[str]:
    from refdist.analysis import hypotheses as H, tables
    from refdist.metrics import flags as fl
    out.mkdir(parents=True, exist_ok=True)
    scores, docs = tables.load()
    cells = tables.cell_table(scores[scores["instance"].isin(fl.load_thresholds())
                                     | ~scores["detector"].str.match(r"^D\d$")], docs)
    made = []
    fig1_h1(cells, out); made.append("fig1")
    fig2_h2(cells, out); made.append("fig2")
    if paths.variants_file().exists() and (scores["group"] == "variants").any():
        fig3_h5(H.h5_deltas(scores, pd.read_parquet(paths.variants_file())), out); made.append("fig3")
    if paths.forecast_file().exists() and cells["stratum"].isin(config.FORECAST_STRATA).any():
        from refdist.analysis.forecast import reveal
        fig4_forecast(reveal(), out); made.append("fig4")
    (out / "README.json").write_text(json.dumps({"figures": made}, indent=1))
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    print(make_all(a.out or paths.results() / "figures"))
