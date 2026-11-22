#!/usr/bin/env python
"""Combine benchmark_kang.py runs on several held-out cell types into one table and one figure.

Each input directory is the --out of one benchmark_kang.py run (it holds metrics.csv and run_log.json).

Usage:
  python examples/bulk/summarize_holdouts.py outputs/bulk_benchmark outputs/bulk_benchmark_B_cells ... \
      --out outputs/bulk_benchmark_summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["sc", "anchored", "homogeneous", "bulk_scaled", "bulk_raw", "copy_bulk", "no_change"]
LABEL = {"sc": "Single-cell prompt (reference)", "anchored": "BulkExpand, anchored",
         "homogeneous": "BulkExpand, homogeneous", "bulk_scaled": "Bulk rescaled to cell depth",
         "bulk_raw": "Raw bulk (no adapter)", "copy_bulk": "Copy the bulk effect (no model)",
         "no_change": "Predict no change (no model)"}
METRICS = ["r_all", "r_top200", "direction_acc_top200", "mae_top200"]


def load(dirs) -> pd.DataFrame:
    frames = []
    for d in map(Path, dirs):
        m = pd.read_csv(d / "metrics.csv")
        info = json.load(open(d / "run_log.json"))
        m.insert(0, "holdout", info["holdout"])
        m["n_query"], m["n_truth"], m["n_genes_scored"] = info["n_query"], info["n_truth"], info["n_genes_scored"]
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    w = df.pivot(index="variant", columns="holdout", values=metric)
    w = w.loc[[v for v in ORDER if v in w.index], list(dict.fromkeys(df["holdout"]))]
    w["mean"] = w.mean(axis=1)
    w.index = [LABEL[v] for v in w.index]
    return w


def plot(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
    color = {"sc": "#2b2a28", "anchored": "#2a78d6", "homogeneous": "#2a78d6", "bulk_scaled": "#b9b7b0",
             "bulk_raw": "#b9b7b0", "copy_bulk": "#eb6834", "no_change": "#dcdad4"}
    holdouts = list(dict.fromkeys(df["holdout"]))
    order = [v for v in ORDER if v in set(df["variant"])]
    panels = [("r_all", "Match to the real response\n(correlation, all genes)", 1.0),
              ("direction_acc_top200", "Up/down direction correct\n(share of 200 most-changed genes)", 1.0),
              ("mae_top200", "Average error, 200 most-changed genes\n(lower is better)", None)]
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), sharey=True)
    H = fig.get_figheight()
    fig.text(0.01, 1 - 0.2 / H, "STACK with single-cell vs bulk prompts, across held-out cell types",
             fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 1 - 0.5 / H,
             f"Kang IFN-beta PBMCs. Each run holds out one cell type ({', '.join(holdouts)}) and predicts its response "
             "from the other cell types.", fontsize=9.5, color=INK2, va="top")
    fig.legend(handles=[Patch(color="#8a8984", label=f"Bar: mean of {len(holdouts)} held-out cell types"),
                        Line2D([], [], marker="o", ls="", color=INK, markeredgecolor=SURF, markersize=7,
                               label="Dot: one held-out cell type")],
               loc="upper left", bbox_to_anchor=(0.006, 1 - 0.74 / H), ncol=2, frameon=False, fontsize=9.5,
               borderaxespad=0)
    y = np.arange(len(order))
    for ax, (col, title, xmax) in zip(axes, panels):
        vals = df.pivot(index="variant", columns="holdout", values=col).loc[order, holdouts].to_numpy(dtype=float)
        has = ~np.all(np.isnan(vals), axis=1)
        means = np.full(len(order), np.nan)
        means[has] = np.nanmean(vals[has], axis=1)
        if xmax is None:
            xmax = np.nanmax(vals) * 1.22
        ax.barh(y, np.clip(np.nan_to_num(means), 0, None), color=[color[v] for v in order], height=0.62,
                edgecolor=SURF)
        for yi in y:
            row = vals[yi][~np.isnan(vals[yi])]
            if len(row):
                jitter = np.linspace(-0.17, 0.17, len(row)) if len(row) > 1 else np.zeros(1)
                ax.scatter(row, yi + jitter, s=26, color=INK, edgecolor=SURF, linewidth=1.0, zorder=3)
                ax.text(max(row.max(), means[yi]) + 0.025 * xmax, yi, f"{means[yi]:.2f}", va="center",
                        fontsize=9, color=INK2)
            else:
                ax.text(0.015 * xmax, yi, "n/a", va="center", fontsize=9, color=INK2)
        ax.set_xlim(0, xmax * (1.08 if col != "mae_top200" else 1.0))
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL[v] for v in order], color=INK)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.2, right=0.98, top=1 - 1.6 / H, bottom=0.09, wspace=0.12)
    fig.savefig(path, dpi=200)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="benchmark_kang.py output directories")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.dirs)
    df.to_csv(out / "all_runs.csv", index=False)
    sizes = df.drop_duplicates("holdout")[["holdout", "n_query", "n_truth", "n_genes_scored"]]
    print(sizes.to_string(index=False))
    for metric in METRICS:
        w = wide(df, metric)
        w.round(3).to_csv(out / f"summary_{metric}.csv")
        print(f"\n{metric}\n{w.round(3).to_string()}")
    plot(df, out / "bulk_vs_sc_kang_holdouts.png")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
