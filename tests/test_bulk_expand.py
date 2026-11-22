"""Tests for stack.bulk (BulkExpand). They use simulated data with a known effect; no checkpoint needed."""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from stack.bulk import (  # noqa: E402
    ExpansionConfig,
    bulk_fold_changes,
    expand_anchored,
    expand_homogeneous,
    pseudobulk,
    to_cpm,
)

N_GENES = 1500
UP, DOWN = np.arange(0, 20), np.arange(20, 40)


def _simulate(seed=0):
    """Atlas of 2 cell types + bulk ctrl/stim from 4 donors with a known fold change."""
    rng = np.random.default_rng(seed)
    genes = [f"Gene{i}" for i in range(N_GENES)]
    base = rng.gamma(0.6, 1.0, size=(2, N_GENES)) + 0.02
    base[:, UP] += 0.5  # effect genes are expressed in both types
    base[:, DOWN] += 0.5
    true_fc = np.ones(N_GENES)
    true_fc[UP], true_fc[DOWN] = 4.0, 0.25

    def cells(n, fc, ctype_p=0.5):
        types = rng.random(n) < ctype_p
        p = np.where(types[:, None], base[0], base[1]) * fc
        p /= p.sum(axis=1, keepdims=True)
        lib = np.exp(rng.normal(np.log(2000), 0.4, size=n))
        return rng.poisson(p * lib[:, None] * rng.gamma(5, 0.2, size=(n, 1)))

    atlas = ad.AnnData(X=sp.csr_matrix(cells(600, np.ones(N_GENES)).astype(np.float32)),
                       var=pd.DataFrame(index=genes))
    rows, names, conds = [], [], []
    for d in range(4):
        for cond, fc in [("ctrl", np.ones(N_GENES)), ("stim", true_fc)]:
            rows.append(cells(800, fc).sum(axis=0))
            names.append(f"donor{d}_{cond}")
            conds.append(cond)
    bulk = pd.DataFrame(np.vstack(rows), index=names, columns=genes)
    design = pd.Series(conds, index=names)
    return atlas, bulk, design, true_fc


def _mean_log_ratio(prompt, cond_a="stim", cond_b="ctrl"):
    X = prompt.X.toarray().astype(float)
    lib = X.sum(axis=1, keepdims=True)
    cp = X / lib * 1e4
    a = cp[prompt.obs["condition"].values == cond_a].mean(axis=0)
    b = cp[prompt.obs["condition"].values == cond_b].mean(axis=0)
    return np.log2((a + 0.5) / (b + 0.5))


def test_bulk_fold_changes_recover_injected_effect():
    _, bulk, design, true_fc = _simulate()
    fcs = bulk_fold_changes(to_cpm(bulk), design, "ctrl", per_replicate=False)
    lfc = np.log2(fcs["stim"][0])
    assert np.all(np.abs(lfc[UP] - 2.0) < 0.5)
    assert np.all(np.abs(lfc[DOWN] + 2.0) < 0.5)
    assert np.allclose(fcs["ctrl"], 1.0)


def test_length_bias_cancels_in_fold_changes():
    _, bulk, design, _ = _simulate()
    lengths = np.random.default_rng(1).uniform(500, 10000, size=N_GENES)
    biased = bulk * (lengths / lengths.mean())  # full-length RNA-seq reads scale with length
    a = np.log2(bulk_fold_changes(to_cpm(bulk), design, "ctrl", per_replicate=False)["stim"][0])
    b = np.log2(bulk_fold_changes(to_cpm(biased), design, "ctrl", per_replicate=False)["stim"][0])
    assert np.corrcoef(a, b)[0, 1] > 0.99
    assert np.all(np.sign(b[UP]) > 0) and np.all(np.sign(b[DOWN]) < 0)


def test_anchored_output_schema():
    atlas, bulk, design, _ = _simulate()
    cfg = ExpansionConfig(n_cells_per_condition=200, seed=0)
    prompt = expand_anchored(atlas, bulk, design, "ctrl", cfg)
    assert prompt.n_obs == 400
    assert set(prompt.obs["condition"]) == {"ctrl", "stim"}
    assert (prompt.obs["organism"] == "Homo sapiens").all()
    assert prompt.obs_names.is_unique
    assert list(prompt.var_names) == list(atlas.var_names)
    X = prompt.X.toarray()
    assert np.issubdtype(X.dtype, np.integer) and X.min() >= 0
    assert set(prompt.obs.loc[prompt.obs["condition"] == "stim", "bulk_replicate"]) <= set(design.index)


def test_anchored_matches_single_cell_depth_and_sparsity():
    atlas, bulk, design, _ = _simulate()
    prompt = expand_anchored(atlas, bulk, design, "ctrl", ExpansionConfig(n_cells_per_condition=400))
    ctrl = prompt[prompt.obs["condition"] == "ctrl"].X.toarray()
    atlas_lib = np.asarray(atlas.X.sum(axis=1)).ravel()
    ratio = np.median(ctrl.sum(axis=1)) / np.median(atlas_lib)
    assert 0.85 < ratio < 1.15  # single-cell depth, not bulk depth
    zero_atlas = (atlas.X.toarray() == 0).mean()
    zero_ctrl = (ctrl == 0).mean()
    assert zero_ctrl > 0.3 and abs(zero_ctrl - zero_atlas) < 0.1  # as sparse as the real cells


def test_anchored_transfers_bulk_effect():
    atlas, bulk, design, true_fc = _simulate()
    prompt = expand_anchored(atlas, bulk, design, "ctrl", ExpansionConfig(n_cells_per_condition=600))
    lr = _mean_log_ratio(prompt)
    assert np.corrcoef(lr, np.log2(true_fc))[0, 1] > 0.9
    assert (lr[UP] > 0.5).all() and (lr[DOWN] < -0.5).all()
    others = np.setdiff1d(np.arange(N_GENES), np.r_[UP, DOWN])
    assert np.median(np.abs(lr[others])) < 0.35  # unaffected genes stay roughly unchanged


def test_homogeneous_mode_depth_sparsity_and_effect():
    atlas, bulk, design, true_fc = _simulate()
    libs = np.asarray(atlas.X.sum(axis=1)).ravel()
    prompt = expand_homogeneous(bulk, design, ExpansionConfig(n_cells_per_condition=500), library_sizes=libs)
    X = prompt.X.toarray()
    assert 0.8 < np.median(X.sum(axis=1)) / np.median(libs) < 1.25
    zero_atlas = (atlas.X.toarray() == 0).mean()
    assert (X == 0).mean() > 0.3 and abs((X == 0).mean() - zero_atlas) < 0.15
    lr = _mean_log_ratio(prompt)
    assert (lr[UP] > 0.5).all() and (lr[DOWN] < -0.5).all()


def test_theta_series_is_aligned_by_gene_name():
    _, bulk, design, _ = _simulate()
    theta = pd.Series(1.0, index=[g.lower() for g in bulk.columns])
    prompt = expand_homogeneous(bulk, design, ExpansionConfig(n_cells_per_condition=50), theta=theta)
    assert prompt.n_obs == 100


def test_seed_makes_output_deterministic():
    atlas, bulk, design, _ = _simulate()
    cfg = ExpansionConfig(n_cells_per_condition=100, seed=7)
    a = expand_anchored(atlas, bulk, design, "ctrl", cfg).X.toarray()
    b = expand_anchored(atlas, bulk, design, "ctrl", cfg).X.toarray()
    assert np.array_equal(a, b)


def test_duplicate_gene_symbols_are_dropped():
    atlas, bulk, design, _ = _simulate()
    names = list(atlas.var_names)
    names[1] = names[0].lower()  # 'gene0' collides with 'Gene0' after upper-casing
    atlas.var_names = names
    prompt = expand_anchored(atlas, bulk, design, "ctrl", ExpansionConfig(n_cells_per_condition=20))
    assert pd.Index([g.upper() for g in prompt.var_names]).is_unique


def test_pseudobulk_sums_groups():
    X = sp.csr_matrix(np.array([[1, 0], [2, 3], [4, 5]]))
    pb = pseudobulk(X, ["a", "b", "a"])
    assert pb.loc["a"].tolist() == [5, 5] and pb.loc["b"].tolist() == [2, 3]


def test_cli_end_to_end(tmp_path):
    from stack.bulk.cli import main

    atlas, bulk, design, _ = _simulate()
    bulk.T.to_csv(tmp_path / "bulk.tsv", sep="\t")  # GEO style: genes in rows
    pd.DataFrame({"sample": design.index, "condition": design.values}).to_csv(tmp_path / "design.csv", index=False)
    atlas.write_h5ad(tmp_path / "atlas.h5ad")
    out = tmp_path / "prompt.h5ad"
    main(["--bulk", str(tmp_path / "bulk.tsv"), "--design", str(tmp_path / "design.csv"), "--control", "ctrl",
          "--atlas", str(tmp_path / "atlas.h5ad"), "--n-cells", "50", "--out", str(out)])
    prompt = ad.read_h5ad(out)
    assert prompt.n_obs == 100 and prompt.uns["bulk_expand"]["mode"] == "anchored"
    out2 = tmp_path / "prompt_h.h5ad"
    main(["--bulk", str(tmp_path / "bulk.tsv"), "--design", str(tmp_path / "design.csv"), "--control", "ctrl",
          "--n-cells", "30", "--out", str(out2)])
    assert ad.read_h5ad(out2).uns["bulk_expand"]["mode"] == "homogeneous"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
