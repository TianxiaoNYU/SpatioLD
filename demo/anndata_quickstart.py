"""AnnData integration demo for SpatioLD."""

from __future__ import annotations

import numpy as np
import anndata as ad

from spatiold import SpatioLD


adata = ad.AnnData(X=np.empty((8, 0)))
adata.obs_names = [f"cell_{i}" for i in range(8)]
adata.obs["cell_type"] = ["A", "A", "A", "B", "B", "B", "C", "C"]
adata.obsm["spatial"] = np.array(
    [
        [0.0, 0.0],
        [0.2, 0.1],
        [0.4, 0.1],
        [1.0, 0.0],
        [1.2, 0.1],
        [1.4, 0.1],
        [0.8, 0.8],
        [1.0, 0.9],
    ],
    dtype=float,
)

obj = SpatioLD.from_anndata(adata, label_key="cell_type", spatial_key="spatial")

ld_df = obj.compute_local_diversity(radii=[0.25, 1.2], key="ld_demo")
print("Local diversity:")
print(ld_df.round(3))

perm_stats = obj.compute_permutation_stats(
    n_perm=100,
    radii=[1.2],
    random_state=11,
    n_jobs=1,
    pvals_key="ld_pvals_demo",
    perm_mean_key="ld_perm_mean_demo",
)
pval_df = perm_stats["pvals"]
perm_mean_df = perm_stats["perm_mean"]
perm_dist = perm_stats["distribution"]

print("\nPermutation p-values:")
print(pval_df.round(3))
print("\nPermutation null mean:")
print(perm_mean_df.round(3))
print(f"\nPermutation distribution shape: {perm_dist.shape}")

# Recover stored matrix from adata.obsm through SpatioLD helper
reloaded = obj.get_result("ld_demo")
print("\nReloaded from adata:")
print(reloaded.round(3))
