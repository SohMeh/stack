#!/usr/bin/env python
"""Can STACK use bulk reference perturbations? Benchmark on Kang et al. 2018 (IFN-beta, human PBMCs).

Task: predict how untreated CD14+ monocytes respond to IFN-beta. STACK is shown the perturbation
only in the *other* cell types (the reference context). The real treated monocytes are held out
as the answer key and are never shown to STACK.

The prompt carries the same biology in five formats:
  sc           real single cells, untreated + treated (upper reference)
  bulk_raw     per-donor pseudobulk fed directly as if each were a cell (what happens today)
  bulk_scaled  per-donor pseudobulk rescaled to single-cell depth (right scale, but dense)
  homogeneous  BulkExpand, homogeneous mode: single-cell depth + sparsity from the bulk profile
  anchored     BulkExpand, anchored mode: real untreated cells + the bulk fold change
No-model baselines: "copy the bulk effect" and "predict no change".

Usage (STACK environment):
  python examples/bulk/benchmark_kang.py --kang kang_2018.h5ad --checkpoint bc_large_aligned.ckpt \
      --genelist basecount_1000per_15000max.pkl --out bulk_benchmark/
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from stack.bulk import ExpansionConfig, estimate_nb_theta, expand_anchored, expand_homogeneous, pseudobulk  # noqa: E402

VARIANTS = ["sc", "anchored", "homogeneous", "bulk_scaled", "bulk_raw"]
LABEL = {"sc": "Single-cell prompt (reference)", "anchored": "BulkExpand, anchored",
         "homogeneous": "BulkExpand, homogeneous", "bulk_scaled": "Bulk rescaled to cell depth",
         "bulk_raw": "Raw bulk (no adapter)", "copy_bulk": "Copy the bulk effect (no model)",
         "no_change": "Predict no change (no model)"}


def lognorm(X) -> np.ndarray:
    X = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float)
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1
    return np.log1p(X / lib * 1e4)


def subsample(adata, n, rng):
    if adata.n_obs <= n:
        return adata.copy()
    return adata[np.sort(rng.choice(adata.n_obs, n, replace=False))].copy()


def prompt_from_matrix(X, conditions, genes, tag) -> ad.AnnData:
    a = ad.AnnData(X=sp.csr_matrix(np.asarray(X, dtype=np.float32)),
                   obs=pd.DataFrame({"condition": list(conditions)}),
                   var=pd.DataFrame(index=pd.Index(genes, dtype=str)))
    a.obs_names = [f"{tag}_{i}" for i in range(a.n_obs)]
    a.obs["organism"] = "Homo sapiens"
    return a


def run_stack(prompt_path, query_path, outdir, args) -> dict:
    exe = shutil.which("stack-generation") or str(Path(sys.executable).parent / "stack-generation")
    cmd = [exe, "--checkpoint", args.checkpoint, "--genelist", args.genelist, "--base-adata", str(prompt_path),
           "--test-adata", str(query_path), "--split-column", "condition", "--output-dir", str(outdir),
           "--device", args.device, "--batch-size", "16", "--num-workers", "0", "--num-steps", "5",
           "--prompt-ratio", "0.25", "--context-ratio", "0.4", "--random-seed", str(args.seed)]
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    return {"exit_code": rc, "seconds": round(time.time() - t0, 1)}


def score(pred_effect, real_effect, top):
    """Correlation, up/down direction accuracy and error of a predicted effect vs the real one."""
    nan = dict(r_all=np.nan, r_top200=np.nan, direction_acc_top200=np.nan, mae_top200=np.nan)
    if pred_effect is None or not np.all(np.isfinite(pred_effect)):
        return nan
    mae = float(np.abs(pred_effect[top] - real_effect[top]).mean())
    if pred_effect.std() == 0:  # e.g. "predict no change": correlation and direction are undefined
        return {**nan, "mae_top200": mae}
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    return dict(r_all=r(pred_effect, real_effect), r_top200=r(pred_effect[top], real_effect[top]),
                direction_acc_top200=float((np.sign(pred_effect[top]) == np.sign(real_effect[top])).mean()),
                mae_top200=mae)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kang", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--genelist", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--holdout", default="CD14+ Monocytes")
    p.add_argument("--max-prompt-cells", type=int, default=2000, help="per condition, single-cell prompt")
    p.add_argument("--max-query-cells", type=int, default=1500)
    p.add_argument("--n-synthetic", type=int, default=1000, help="BulkExpand cells per condition")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variants", nargs="*", default=VARIANTS)
    args = p.parse_args()

    out = Path(args.out); (out / "inputs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    kang = ad.read_h5ad(args.kang)
    kang.obs_names_make_unique()
    genes = list(kang.var_names)
    is_ho = (kang.obs["cell_type"] == args.holdout).values
    ctrl = (kang.obs["label"] == "ctrl").values

    query = subsample(kang[is_ho & ctrl], args.max_query_cells, rng)
    truth = kang[is_ho & ~ctrl].copy()
    other = kang[~is_ho].copy()
    query.obs["organism"] = "Homo sapiens"
    query.write_h5ad(out / "inputs" / "query.h5ad")
    print(f"query {query.n_obs} untreated {args.holdout} | answer key {truth.n_obs} treated | "
          f"reference context {other.n_obs} cells of other types")

    # bulk reference: one pseudobulk sample per donor x condition from ALL reference-context cells
    groups = other.obs["replicate"].astype(str) + "__" + other.obs["label"].astype(str)
    pb = pseudobulk(other.X, groups.values)
    pb.columns = genes
    design = pd.Series([s.split("__")[1] for s in pb.index], index=pb.index)
    sc_lib = np.asarray(other.X.sum(axis=1)).ravel()
    print(f"pseudobulk: {pb.shape[0]} samples, median depth {np.median(pb.sum(axis=1)):,.0f} vs "
          f"median single-cell depth {np.median(sc_lib):,.0f}")

    atlas = subsample(other[other.obs["label"] == "ctrl"], 3000, rng)  # unperturbed reference-context cells
    cfg = ExpansionConfig(n_cells_per_condition=args.n_synthetic, seed=args.seed)
    prompts = {}
    if "sc" in args.variants:
        sc_prompt = ad.concat([subsample(other[other.obs["label"] == lab], args.max_prompt_cells, rng)
                               for lab in ["ctrl", "stim"]])
        sc_prompt.obs["condition"] = sc_prompt.obs["label"].astype(str).values
        sc_prompt.obs["organism"] = "Homo sapiens"
        sc_prompt.obs_names_make_unique()
        prompts["sc"] = sc_prompt
    if "bulk_raw" in args.variants:
        prompts["bulk_raw"] = prompt_from_matrix(pb.values, design.values, genes, "bulkraw")
    if "bulk_scaled" in args.variants:
        scaled = pb.values / pb.values.sum(axis=1, keepdims=True) * np.median(sc_lib)
        prompts["bulk_scaled"] = prompt_from_matrix(scaled, design.values, genes, "bulkscaled")
    if "homogeneous" in args.variants:
        theta = pd.Series(estimate_nb_theta(atlas.X), index=atlas.var_names.astype(str))
        prompts["homogeneous"] = expand_homogeneous(pb, design, cfg, library_sizes=sc_lib, theta=theta)
    if "anchored" in args.variants:
        prompts["anchored"] = expand_anchored(atlas, pb, design, "ctrl", cfg)

    # ground truth effect and no-model baselines, on genes present everywhere
    log_q, log_t = lognorm(query.X), lognorm(truth.X)
    real_full = log_t.mean(0) - log_q.mean(0)
    pbn = np.log1p(pb.values / pb.values.sum(axis=1, keepdims=True) * 1e4)
    copy_bulk_full = pbn[(design == "stim").values].mean(0) - pbn[(design == "ctrl").values].mean(0)

    rows, log = [], {}
    gidx = pd.Series(np.arange(len(genes)), index=pd.Index([g.upper() for g in genes]))
    for v, prompt in prompts.items():
        ppath = out / "inputs" / f"prompt_{v}.h5ad"
        prompt.write_h5ad(ppath)
        print(f"\n=== {v}: {prompt.n_obs} prompt cells, median depth {np.median(np.asarray(prompt.X.sum(axis=1)).ravel()):,.0f}")
        info = run_stack(ppath, out / "inputs" / "query.h5ad", out / f"pred_{v}", args)
        log[v] = info
        eff, n_genes = None, 0
        try:
            pc = ad.read_h5ad(out / f"pred_{v}" / "ctrl.h5ad")
            ps = ad.read_h5ad(out / f"pred_{v}" / "stim.h5ad")
            shared = [g for g in pc.var_names if g.upper() in gidx.index]
            cols = gidx.reindex([g.upper() for g in shared]).to_numpy()
            pe = lognorm(ps[:, shared].X).mean(0) - lognorm(pc[:, shared].X).mean(0)
            eff = np.full(len(genes), np.nan); eff[cols] = pe; n_genes = len(shared)
        except Exception as e:  # a failed run is a result too
            print(f"   could not read predictions for {v}: {e}")
        log[v]["n_genes_scored"] = n_genes
        rows.append((v, eff))

    # score every method on the same gene set (genes STACK returned for all successful variants)
    ok = np.ones(len(genes), dtype=bool)
    for _, eff in rows:
        if eff is not None:
            ok &= np.isfinite(eff)
    real_ok = real_full[ok]
    top = np.argsort(-np.abs(real_ok))[:200]  # the 200 genes that changed most in real monocytes
    metrics = []
    for v, eff in rows + [("copy_bulk", copy_bulk_full), ("no_change", np.zeros(len(genes)))]:
        m = score(None if eff is None else eff[ok], real_ok, top)
        metrics.append({"variant": v, "label": LABEL[v], **m,
                        **({k: log[v][k] for k in ("exit_code", "seconds")} if v in log else {})})
    M = pd.DataFrame(metrics)
    M.to_csv(out / "metrics.csv", index=False)
    json.dump({"holdout": args.holdout, "n_query": int(query.n_obs), "n_truth": int(truth.n_obs),
               "n_genes_scored": int(ok.sum()), "runs": log}, open(out / "run_log.json", "w"), indent=2)
    print("\n" + M.round(3).to_string(index=False))
    try:
        plot(M, out / "bulk_vs_sc_kang.png", int(ok.sum()), args.holdout)
    except Exception as e:
        print("plot failed:", e)


def plot(M, path, n_genes, holdout):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
    color = {"sc": "#2b2a28", "anchored": "#2a78d6", "homogeneous": "#2a78d6", "bulk_scaled": "#b9b7b0",
             "bulk_raw": "#b9b7b0", "copy_bulk": "#eb6834", "no_change": "#dcdad4"}
    order = [v for v in ["sc", "anchored", "homogeneous", "bulk_scaled", "bulk_raw", "copy_bulk", "no_change"]
             if v in set(M["variant"])]
    M = M.set_index("variant").loc[order]
    panels = [("r_top200", "Match to the real response\n(correlation, 200 most-changed genes)", (0, 1)),
              ("direction_acc_top200", "Up/down direction correct\n(share of 200 most-changed genes)", (0, 1)),
              ("mae_top200", "Average error on those genes\n(lower is better)", None)]
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    H = fig.get_figheight()
    fig.text(0.01, 1 - 0.2 / H, "Can STACK use bulk reference perturbations? Single-cell vs bulk prompts",
             fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 1 - 0.5 / H, f"Kang IFN-beta PBMCs. Predict held-out {holdout} from the other cell types' response. "
             f"Bulk = per-donor pseudobulk of those same cells. {n_genes:,} genes scored.",
             fontsize=9.5, color=INK2, va="top")
    y = np.arange(len(order))
    for ax, (col, title, lim) in zip(axes, panels):
        vals = M[col].to_numpy(dtype=float)
        ax.barh(y, np.clip(np.nan_to_num(vals), 0, None), color=[color[v] for v in order], height=0.6, edgecolor=SURF)
        xmax = lim[1] if lim else np.nanmax(vals) * 1.25
        for yi, v in zip(y, vals):
            name = order[int(yi)]
            txt = f"{v:.2f}" if not np.isnan(v) else ("n/a" if name == "no_change" else "failed")
            ax.text((0 if np.isnan(v) else max(v, 0)) + 0.015 * xmax, yi, txt, va="center", fontsize=9, color=INK2)
        ax.set_xlim(0, xmax)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
        ax.grid(axis="x", color=GRID); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y); axes[0].set_yticklabels([LABEL[v] for v in order], color=INK)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.2, right=0.98, top=1 - 1.25 / H, bottom=0.12, wspace=0.12)
    fig.savefig(path, dpi=200)
    print("wrote", path)


if __name__ == "__main__":
    main()
