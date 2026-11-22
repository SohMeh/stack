"""``stack-bulk-prompt``: build a STACK prompt .h5ad from bulk RNA-seq.

Example (anchored, recommended)::

    stack-bulk-prompt --bulk bulk_counts.tsv --design design.csv --control DMSO \\
        --atlas unperturbed_cells.h5ad --out prompt.h5ad

    stack-generation --checkpoint bc_large_aligned.ckpt --genelist basecount_1000per_15000max.pkl \\
        --base-adata prompt.h5ad --split-column condition --test-adata query.h5ad --output-dir out/

Bulk table: genes x samples (GEO style) or samples x genes, as .csv/.tsv/.txt(.gz) or .h5ad.
Design: .csv/.tsv with columns ``sample`` and ``condition``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

from .expand import ExpansionConfig, estimate_nb_theta, expand_anchored, expand_homogeneous

LOGGER = logging.getLogger("stack.bulk")


def _read_table(path: str) -> pd.DataFrame:
    sep = "," if ".csv" in Path(path).name.lower() else "\t"
    return pd.read_csv(path, sep=sep, index_col=0)


def load_bulk(path: str, genes_in: str = "auto") -> pd.DataFrame:
    """Return bulk counts as samples x genes."""
    if path.endswith(".h5ad"):
        a = ad.read_h5ad(path)
        X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        return pd.DataFrame(X, index=a.obs_names, columns=a.var_names)
    df = _read_table(path)
    df = df.select_dtypes(include=[np.number])
    if genes_in == "rows" or (genes_in == "auto" and df.shape[0] > df.shape[1]):
        df = df.T
    return df


def load_design(path: str) -> pd.Series:
    sep = "," if ".csv" in Path(path).name.lower() else "\t"
    d = pd.read_csv(path, sep=sep)
    cols = {c.lower(): c for c in d.columns}
    if "sample" not in cols or "condition" not in cols:
        raise SystemExit("design file needs columns 'sample' and 'condition'")
    return pd.Series(d[cols["condition"]].astype(str).values, index=d[cols["sample"]].astype(str).values)


def load_lengths(path: Optional[str]) -> Optional[pd.Series]:
    if not path:
        return None
    df = _read_table(path)
    return df.iloc[:, 0].astype(float)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert bulk RNA-seq into a synthetic single-cell STACK prompt.")
    p.add_argument("--bulk", required=True, help="bulk counts table or .h5ad")
    p.add_argument("--design", required=True, help="table with columns sample, condition")
    p.add_argument("--control", required=True, help="control condition label (anchored mode)")
    p.add_argument("--out", required=True, help="output prompt .h5ad")
    p.add_argument("--atlas", help="unperturbed single-cell .h5ad of the reference context (raw counts)")
    p.add_argument("--mode", choices=["auto", "anchored", "homogeneous"], default="auto",
                   help="auto = anchored when --atlas is given, homogeneous otherwise")
    p.add_argument("--genes-in", choices=["auto", "rows", "columns"], default="auto")
    p.add_argument("--gene-lengths", help="optional table gene -> length (bp); length-normalizes full-length bulk")
    p.add_argument("--n-cells", type=int, default=500, help="synthetic cells per condition")
    p.add_argument("--pseudocount-cpm", type=float, default=1.0)
    p.add_argument("--max-fold-change", type=float, default=32.0)
    p.add_argument("--no-per-replicate", action="store_true", help="use the mean fold change for every cell")
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--blend", type=float, default=0.02)
    p.add_argument("--library-size-median", type=float, default=3000.0)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args(argv)
    bulk = load_bulk(args.bulk, args.genes_in)
    design = load_design(args.design)
    bulk.index = bulk.index.astype(str)
    missing = sorted(set(bulk.index) - set(design.index))
    if missing:
        LOGGER.warning("Dropping %d bulk samples without a condition: %s", len(missing), missing[:5])
        bulk = bulk.loc[bulk.index.isin(design.index)]
    cfg = ExpansionConfig(
        n_cells_per_condition=args.n_cells, pseudocount_cpm=args.pseudocount_cpm,
        max_fold_change=args.max_fold_change, per_replicate=not args.no_per_replicate,
        n_neighbors=args.n_neighbors, blend=args.blend, library_size_median=args.library_size_median,
        seed=args.seed,
    )
    lengths = load_lengths(args.gene_lengths)
    mode = args.mode if args.mode != "auto" else ("anchored" if args.atlas else "homogeneous")
    LOGGER.info("bulk: %d samples x %d genes | conditions: %s | mode: %s",
                bulk.shape[0], bulk.shape[1], design.reindex(bulk.index).value_counts().to_dict(), mode)
    if mode == "anchored":
        if not args.atlas:
            raise SystemExit("--mode anchored needs --atlas")
        atlas = ad.read_h5ad(args.atlas)
        prompt = expand_anchored(atlas, bulk, design, args.control, cfg, gene_lengths=lengths)
    else:
        libs = theta = None
        if args.atlas:
            atlas = ad.read_h5ad(args.atlas)
            libs = np.asarray(atlas.X.sum(axis=1)).ravel()
            theta = pd.Series(estimate_nb_theta(atlas.X), index=atlas.var_names.astype(str))
        prompt = expand_homogeneous(bulk, design, cfg, library_sizes=libs, theta=theta, gene_lengths=lengths)
    prompt.uns["bulk_expand"] = json.loads(json.dumps(prompt.uns["bulk_expand"], default=str))
    prompt.write_h5ad(args.out)
    LOGGER.info("wrote %s: %d synthetic cells x %d genes; conditions %s", args.out, prompt.n_obs, prompt.n_vars,
                prompt.obs["condition"].value_counts().to_dict())


if __name__ == "__main__":
    main()
