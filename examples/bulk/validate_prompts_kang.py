#!/usr/bin/env python
"""Do BulkExpand's synthetic cells look like real cells? (No STACK needed.)

Builds the same Kang reference context as benchmark_kang.py, turns its per-donor pseudobulk into
synthetic prompt cells, and compares them with the real single cells:
  depth and sparsity, agreement of the treated-minus-control effect with the real cells' effect,
  and whether a classifier on 30 PCs can tell synthetic untreated cells from real ones (AUC 0.5 = cannot).
Needs scikit-learn.  Usage: python examples/bulk/validate_prompts_kang.py kang_2018.h5ad
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from stack.bulk import ExpansionConfig, estimate_nb_theta, expand_anchored, expand_homogeneous, pseudobulk
warnings.filterwarnings("error", category=RuntimeWarning)  # any FP warning in our code = failure here

kang = ad.read_h5ad(sys.argv[1]); kang.obs_names_make_unique()
rng = np.random.default_rng(0)
ho = (kang.obs["cell_type"] == "CD14+ Monocytes").values
other = kang[~ho].copy()
lab = other.obs["label"].astype(str).values
def sub(a, n):
    return a[np.sort(rng.choice(a.n_obs, min(n, a.n_obs), replace=False))].copy()
groups = other.obs["replicate"].astype(str) + "__" + other.obs["label"].astype(str)
pb = pseudobulk(other.X, groups.values); pb.columns = list(kang.var_names)
design = pd.Series([s.split("__")[1] for s in pb.index], index=pb.index)
atlas = sub(other[lab == "ctrl"], 3000)
cfg = ExpansionConfig(n_cells_per_condition=1000, seed=0)
t = time.time(); anch = expand_anchored(atlas, pb, design, "ctrl", cfg); ta = time.time() - t
theta = pd.Series(estimate_nb_theta(atlas.X), index=atlas.var_names.astype(str))
homog = expand_homogeneous(pb, design, cfg, library_sizes=np.asarray(other.X.sum(1)).ravel(), theta=theta)
real = ad.concat([sub(other[lab == l], 2000) for l in ["ctrl", "stim"]]); real.obs["condition"] = real.obs["label"].astype(str).values
print(f"anchored expansion took {ta:.1f}s, no floating-point warnings")

genes = list(kang.var_names); up = pd.Index([g.upper() for g in genes])
def ln(X):
    X = X.toarray() if sp.issparse(X) else np.asarray(X, float); L = X.sum(1, keepdims=True); L[L == 0] = 1
    return np.log1p(X / L * 1e4)
def mat(a):  # align to Kang gene order
    idx = pd.Index([g.upper() for g in a.var_names]).get_indexer(up)
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    out = np.zeros((a.n_obs, len(genes))); ok = idx >= 0; out[:, ok] = X[:, idx[ok]]; return out
R = {n: (mat(a), a.obs["condition"].astype(str).values) for n, a in [("real cells", real), ("BulkExpand anchored", anch), ("BulkExpand homogeneous", homog)]}
Xr, cr = R["real cells"]
eff_real = ln(Xr[cr == "stim"]).mean(0) - ln(Xr[cr == "ctrl"]).mean(0)
top = np.argsort(-np.abs(eff_real))[:200]
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
pca = PCA(30, random_state=0).fit(ln(Xr))
rows = []
for n, (X, c) in R.items():
    e = ln(X[c == "stim"]).mean(0) - ln(X[c == "ctrl"]).mean(0)
    row = {"prompt": n, "median_depth": np.median(X.sum(1)), "zero_fraction": (X == 0).mean(),
           "effect_r_all": np.corrcoef(e, eff_real)[0, 1], "effect_r_top200": np.corrcoef(e[top], eff_real[top])[0, 1],
           "direction_top200": (np.sign(e[top]) == np.sign(eff_real[top])).mean()}
    if n != "real cells":  # can a classifier tell synthetic untreated cells from real untreated cells?
        A = pca.transform(ln(X[c == "ctrl"][:1000])); B = pca.transform(ln(Xr[cr == "ctrl"][:1000]))
        y = np.r_[np.ones(len(A)), np.zeros(len(B))]
        row["real_vs_synthetic_auc"] = cross_val_score(LogisticRegression(max_iter=2000), np.vstack([A, B]), y, cv=5, scoring="roc_auc").mean()
    rows.append(row)
pbX = pb.values; d = design.values
row = {"prompt": "raw pseudobulk", "median_depth": np.median(pbX.sum(1)), "zero_fraction": (pbX == 0).mean()}
eb = ln(pbX[d == "stim"]).mean(0) - ln(pbX[d == "ctrl"]).mean(0)
row.update(effect_r_all=np.corrcoef(eb, eff_real)[0, 1], effect_r_top200=np.corrcoef(eb[top], eff_real[top])[0, 1],
           direction_top200=(np.sign(eb[top]) == np.sign(eff_real[top])).mean())
rows.append(row)
print(pd.DataFrame(rows).round(3).to_string(index=False))
