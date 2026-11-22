"""BulkExpand: turn bulk RNA-seq into synthetic single-cell prompts that STACK can read.

Why this is needed
------------------
STACK reads *raw UMI counts* for a *set of cells*. Inside the model the counts go through
``log1p`` with no depth normalization, the decoder scales its output by each cell's library
size, and the across-cell attention compares many individual cells. A bulk RNA-seq sample
breaks all three assumptions: it has millions of reads instead of thousands, almost no zeros,
and it is one averaged vector instead of a population of cells.

What this module does
---------------------
It converts bulk reference perturbations into prompt cells that look statistically like real
single cells, so the unmodified STACK checkpoint can use them in context.

* ``anchored`` mode (recommended). Use an *unperturbed* single-cell atlas of the reference
  context. For every synthetic cell, take a real atlas cell, smooth it over its nearest
  neighbours, multiply it by the per-gene fold change measured in bulk (treated / control),
  and resample UMI counts at that cell's own sequencing depth. Control prompt cells go through
  exactly the same smoothing and resampling with a fold change of 1, so the only systematic
  difference between the control and treated prompts is the bulk-measured effect.
  Because only a *ratio* of two bulk samples is used, bulk-specific biases that affect both
  samples equally (transcript-length bias of full-length RNA-seq, library prep, depth) cancel.
* ``homogeneous`` mode (fallback when no atlas exists). Sample cells directly from the bulk
  profile with realistic single-cell library sizes and negative-binomial noise.

The output is an ordinary AnnData prompt (raw integer counts, one ``condition`` column) that
``stack-generation`` consumes unchanged via ``--base-adata ... --split-column condition``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "ExpansionConfig",
    "to_cpm",
    "bulk_fold_changes",
    "estimate_nb_theta",
    "expand_anchored",
    "expand_homogeneous",
    "pseudobulk",
]


@dataclass
class ExpansionConfig:
    """Settings for bulk-to-cell expansion."""

    n_cells_per_condition: int = 500
    pseudocount_cpm: float = 1.0  # added to bulk CPM before taking ratios; stabilizes low genes
    max_fold_change: float = 32.0  # fold changes are clipped to [1/max, max]
    per_replicate: bool = True  # each synthetic cell carries one bulk replicate's fold change
    n_neighbors: int = 15  # kNN smoothing of atlas cells (anchored mode)
    n_pcs: int = 30
    n_hvg: int = 2000
    blend: float = 0.02  # weight of the population-level target mixed into each cell
    max_atlas_cells: int = 5000
    library_size_median: float = 3000.0  # homogeneous mode without an atlas
    library_size_log_sd: float = 0.5
    theta: float = 2.0  # homogeneous mode: NB inverse dispersion when no atlas is given
    chunk_size: int = 512
    seed: int = 0


# ----------------------------------------------------------------------------- helpers


def _as_dense(x) -> np.ndarray:
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def _upper_index(names: Sequence[str]) -> pd.Index:
    return pd.Index([str(n).upper() for n in names])


def to_cpm(counts: pd.DataFrame, gene_lengths: Optional[pd.Series] = None) -> pd.DataFrame:
    """Samples x genes counts to counts-per-million.

    If ``gene_lengths`` (bp, indexed by gene) is given, counts are divided by length first,
    which gives TPM-like values for full-length bulk RNA-seq.
    """
    x = counts.astype(float)
    if gene_lengths is not None:
        lengths = gene_lengths.reindex(x.columns)
        keep = lengths.notna() & (lengths > 0)
        x = x.loc[:, keep] / (lengths[keep].to_numpy() / 1000.0)
    totals = x.sum(axis=1).replace(0, np.nan)
    return x.div(totals, axis=0).fillna(0.0) * 1e6


def bulk_fold_changes(
    cpm: pd.DataFrame,
    conditions: pd.Series,
    control: str,
    pseudocount_cpm: float = 1.0,
    max_fold_change: float = 32.0,
    per_replicate: bool = True,
) -> Dict[str, np.ndarray]:
    """Per-gene fold changes of each condition against the control mean.

    Returns ``{condition: array (n_replicates or 1, n_genes)}``. The control maps to ones.
    """
    conditions = conditions.reindex(cpm.index)
    if conditions.isna().any():
        missing = list(conditions[conditions.isna()].index[:5])
        raise ValueError(f"No condition given for bulk samples: {missing}")
    if control not in set(conditions):
        raise ValueError(f"Control label '{control}' not found in conditions {sorted(set(conditions))}")
    ctrl_mean = cpm.loc[conditions == control].mean(axis=0).to_numpy()
    out: Dict[str, np.ndarray] = {control: np.ones((1, cpm.shape[1]))}
    for cond in sorted(set(conditions) - {control}):
        reps = cpm.loc[conditions == cond].to_numpy()
        if not per_replicate:
            reps = reps.mean(axis=0, keepdims=True)
        fc = (reps + pseudocount_cpm) / (ctrl_mean[None, :] + pseudocount_cpm)
        out[cond] = np.clip(fc, 1.0 / max_fold_change, max_fold_change)
    return out


def estimate_nb_theta(counts, min_mean: float = 1e-3) -> np.ndarray:
    """Per-gene NB inverse dispersion (theta) by method of moments on depth-normalized counts."""
    X = sp.csr_matrix(counts, dtype=np.float64)
    lib = np.asarray(X.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    scale = np.median(lib) / lib
    Xn = sp.diags(scale) @ X
    mean = np.asarray(Xn.mean(axis=0)).ravel()
    mean_sq = np.asarray(Xn.multiply(Xn).mean(axis=0)).ravel()
    var = mean_sq - mean**2
    excess = var - mean
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.where(excess > 0, mean**2 / excess, 1e4)
    ok = mean > min_mean
    fill = np.median(theta[ok]) if ok.any() else 2.0
    theta = np.where(ok, theta, fill)
    return np.clip(theta, 0.05, 1e4)


def pseudobulk(counts, groups: Sequence) -> pd.DataFrame:
    """Sum single-cell counts into one bulk-like profile per group (rows = groups)."""
    groups = pd.Series(np.asarray(groups))
    X = sp.csr_matrix(counts)
    rows, labels = [], []
    for g, idx in groups.groupby(groups).groups.items():
        rows.append(np.asarray(X[np.asarray(list(idx))].sum(axis=0)).ravel())
        labels.append(g)
    return pd.DataFrame(np.vstack(rows), index=labels)


def _knn_graph(logn: sp.csr_matrix, k: int, n_pcs: int, n_hvg: int, chunk: int = 1024) -> sp.csr_matrix:
    """Row-normalized kNN averaging matrix (each cell averages itself and k-1 neighbours).

    ``logn`` is the sparse log1p(CP10k) matrix; only the most variable genes are densified.
    """
    n = logn.shape[0]
    k = max(1, min(k, n))
    mean = np.asarray(logn.mean(axis=0)).ravel()
    var = np.asarray(logn.multiply(logn).mean(axis=0)).ravel() - mean**2
    hv = np.argsort(-var)[: min(n_hvg, logn.shape[1])]
    dense = _as_dense(logn[:, hv])
    Z = dense - dense.mean(axis=0)
    n_pcs = max(1, min(n_pcs, min(Z.shape) - 1)) if min(Z.shape) > 1 else 1
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    P = (U[:, :n_pcs] * S[:n_pcs]).astype(np.float64)
    sq = (P**2).sum(axis=1)
    idx = np.empty((n, k), dtype=np.int64)
    # Some BLAS builds (e.g. macOS Accelerate) raise spurious floating-point flags in matmul, so
    # silence them and guard the result explicitly instead.
    with np.errstate(all="ignore"):
        for s in range(0, n, chunk):
            d = sq[s : s + chunk, None] + sq[None, :] - 2.0 * P[s : s + chunk] @ P.T
            d[~np.isfinite(d)] = np.inf
            rows_here = np.arange(d.shape[0])
            d[rows_here, s + rows_here] = -np.inf  # every cell is part of its own neighbourhood
            idx[s : s + chunk] = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    return sp.csr_matrix((np.full(n * k, 1.0 / k), (rows, idx.ravel())), shape=(n, n))


def _log_norm(X: sp.csr_matrix) -> sp.csr_matrix:
    """Sparse log1p(counts per 10k); zeros stay zero."""
    lib = np.asarray(X.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    Xn = sp.csr_matrix(sp.diags(1e4 / lib) @ X)
    Xn.data = np.log1p(Xn.data)
    return Xn


def _dedupe_genes(adata: ad.AnnData) -> ad.AnnData:
    """Drop genes whose upper-case symbol repeats (STACK aligns genes case-insensitively)."""
    dup = _upper_index(adata.var_names).duplicated()
    return adata[:, ~dup] if dup.any() else adata


def _to_prompt_adata(blocks, var_names, uns) -> ad.AnnData:
    X = sp.vstack([b[0] for b in blocks]).tocsr().astype(np.int32)
    obs = pd.concat([b[1] for b in blocks], ignore_index=True)
    obs.index = [f"bulkexpand_{i}" for i in range(len(obs))]
    obs["organism"] = "Homo sapiens"  # STACK keeps only rows with this exact label
    out = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(list(var_names), dtype=str)))
    out.uns["bulk_expand"] = uns
    return out


# ----------------------------------------------------------------------------- anchored mode


def expand_anchored(
    atlas: ad.AnnData,
    bulk_counts: pd.DataFrame,
    conditions: pd.Series,
    control: str,
    config: Optional[ExpansionConfig] = None,
    gene_lengths: Optional[pd.Series] = None,
) -> ad.AnnData:
    """Build a synthetic single-cell prompt by applying bulk fold changes to atlas cells.

    Parameters
    ----------
    atlas
        Unperturbed single cells of the reference context, raw counts in ``.X`` and gene
        symbols as ``var_names``.
    bulk_counts
        Bulk counts, samples x genes (gene symbols as columns).
    conditions
        Condition label for every bulk sample (index = sample name).
    control
        Label of the control condition in ``conditions``.
    """
    cfg = config or ExpansionConfig()
    rng = np.random.default_rng(cfg.seed)
    atlas = _dedupe_genes(atlas)

    # 1. align genes: output keeps the atlas genes; genes absent from bulk get fold change 1
    bulk_counts = bulk_counts.copy()
    bulk_counts.columns = _upper_index(bulk_counts.columns)
    bulk_counts = bulk_counts.T.groupby(level=0, sort=False).sum().T  # merge duplicate symbols, keep order
    lengths = None
    if gene_lengths is not None:
        lengths = gene_lengths.copy()
        lengths.index = _upper_index(lengths.index)
        lengths = lengths.groupby(level=0).mean()
    cpm = to_cpm(bulk_counts, lengths)
    fcs = bulk_fold_changes(cpm, conditions, control, cfg.pseudocount_cpm, cfg.max_fold_change, cfg.per_replicate)
    cond_aligned = conditions.reindex(cpm.index)
    rep_names = {c: (list(cpm.index[cond_aligned == c]) if cfg.per_replicate else ["mean"]) for c in fcs}
    rep_names[control] = ["control_mean"]

    atlas_genes = _upper_index(atlas.var_names)
    pos = pd.Series(np.arange(len(cpm.columns)), index=cpm.columns)
    in_bulk = atlas_genes.isin(cpm.columns)
    col = pos.reindex(atlas_genes).to_numpy()

    def full_fc(fc_rows: np.ndarray) -> np.ndarray:
        out = np.ones((fc_rows.shape[0], len(atlas_genes)))
        out[:, in_bulk] = fc_rows[:, col[in_bulk].astype(int)]
        return out

    # 2. atlas preprocessing: subsample, library sizes, kNN-smoothed expression proportions
    X = sp.csr_matrix(atlas.X, dtype=np.float64)
    if X.shape[0] > cfg.max_atlas_cells:
        keep = np.sort(rng.choice(X.shape[0], cfg.max_atlas_cells, replace=False))
        X = X[keep]
    lib = np.asarray(X.sum(axis=1)).ravel()
    good = lib > 0
    X, lib = X[good], lib[good]
    props = sp.diags(1.0 / lib) @ X
    W = _knn_graph(_log_norm(X), cfg.n_neighbors, cfg.n_pcs, cfg.n_hvg)
    pop_mean = np.asarray(props.mean(axis=0)).ravel()

    # 3. expand every condition with identical smoothing + resampling
    blocks = []
    for cond, fc_rows in fcs.items():
        fc_full = full_fc(fc_rows)
        n = cfg.n_cells_per_condition
        anchors = rng.choice(X.shape[0], n, replace=n > X.shape[0])
        reps = rng.integers(0, fc_full.shape[0], size=n)
        targets = pop_mean[None, :] * fc_full
        targets /= targets.sum(axis=1, keepdims=True)
        mats = []
        for s in range(0, n, cfg.chunk_size):
            a, r = anchors[s : s + cfg.chunk_size], reps[s : s + cfg.chunk_size]
            q = _as_dense(W[a] @ props) * fc_full[r]
            q /= q.sum(axis=1, keepdims=True)
            q = (1.0 - cfg.blend) * q + cfg.blend * targets[r]
            counts = rng.poisson(q * lib[a][:, None])
            mats.append(sp.csr_matrix(counts))
        obs = pd.DataFrame({
            "condition": cond,
            "bulk_replicate": [str(rep_names[cond][int(i)]) for i in reps],
            "anchor_cell": anchors,
            "bulk_expand_mode": "anchored",
        })
        blocks.append((sp.vstack(mats), obs))

    uns = {
        "mode": "anchored",
        "config": asdict(cfg),
        "control": control,
        "conditions": sorted(fcs),
        "n_atlas_cells_used": int(X.shape[0]),
        "atlas_genes": int(len(atlas_genes)),
        "atlas_genes_with_bulk_fc": int(in_bulk.sum()),
        "length_normalized": gene_lengths is not None,
    }
    return _to_prompt_adata(blocks, atlas.var_names, uns)


# ----------------------------------------------------------------------------- homogeneous mode


def expand_homogeneous(
    bulk_counts: pd.DataFrame,
    conditions: pd.Series,
    config: Optional[ExpansionConfig] = None,
    library_sizes: Optional[np.ndarray] = None,
    theta=None,
    gene_lengths: Optional[pd.Series] = None,
) -> ad.AnnData:
    """Sample synthetic cells directly from bulk profiles (no atlas available).

    Each synthetic cell picks one bulk replicate of its condition, a single-cell library size
    (from ``library_sizes`` if given, otherwise log-normal around ``library_size_median``),
    and gene counts from a negative binomial with inverse dispersion ``theta``.
    """
    cfg = config or ExpansionConfig()
    rng = np.random.default_rng(cfg.seed)
    bulk_counts = bulk_counts.copy()
    bulk_counts.columns = _upper_index(bulk_counts.columns)
    bulk_counts = bulk_counts.T.groupby(level=0, sort=False).sum().T  # merge duplicate symbols, keep order
    lengths = None
    if gene_lengths is not None:
        lengths = gene_lengths.copy()
        lengths.index = _upper_index(lengths.index)
        lengths = lengths.groupby(level=0).mean()
    props = to_cpm(bulk_counts, lengths) / 1e6
    genes = props.columns
    conditions = conditions.reindex(props.index)
    if theta is None:
        th = np.full(len(genes), cfg.theta)
    elif isinstance(theta, pd.Series):  # per-gene theta indexed by symbol, e.g. from an atlas
        t = pd.Series(theta.to_numpy(dtype=float), index=_upper_index(theta.index)).groupby(level=0).median()
        th = t.reindex(genes).fillna(t.median() if len(t) else cfg.theta).to_numpy()
    else:
        th = np.asarray(theta, dtype=float)
        if th.shape[0] != len(genes):
            raise ValueError("theta must have one value per bulk gene (or be a pd.Series indexed by gene)")

    blocks = []
    for cond in sorted(set(conditions)):
        P = props.loc[conditions == cond].to_numpy()
        n = cfg.n_cells_per_condition
        reps = rng.integers(0, P.shape[0], size=n)
        if library_sizes is not None:
            L = rng.choice(np.asarray(library_sizes, dtype=float), size=n, replace=True)
        else:
            L = np.exp(rng.normal(np.log(cfg.library_size_median), cfg.library_size_log_sd, size=n))
        mats = []
        for s in range(0, n, cfg.chunk_size):
            mu = P[reps[s : s + cfg.chunk_size]] * L[s : s + cfg.chunk_size, None]
            lam = rng.gamma(shape=th[None, :], scale=mu / th[None, :])
            mats.append(sp.csr_matrix(rng.poisson(lam)))
        obs = pd.DataFrame({
            "condition": cond,
            "bulk_replicate": [str(props.index[conditions == cond][i]) for i in reps],
            "bulk_expand_mode": "homogeneous",
        })
        blocks.append((sp.vstack(mats), obs))

    uns = {"mode": "homogeneous", "config": asdict(cfg), "conditions": sorted(set(conditions)),
           "length_normalized": gene_lengths is not None, "theta_source": "given" if theta is not None else "config"}
    return _to_prompt_adata(blocks, genes, uns)
