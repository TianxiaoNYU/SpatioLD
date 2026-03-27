"""Synthetic quickstart demo for SpatioLD."""

from __future__ import annotations

import numpy as np
import pandas as pd

import spatiold as sld


# Synthetic 2D layout and labels
coords = pd.DataFrame(
    {
        "x": [0, 0, 1, 1, 2, 2],
        "y": [0, 1, 0, 1, 0, 1],
    },
    index=[f"cell_{i}" for i in range(6)],
)
labels = pd.Series(["A", "A", "B", "B", "B", "A"], index=coords.index)

radii = [0.2, 1.1, 2.2]

ld_df = sld.compute_local_diversity_multi_radius(coords, labels, radii=radii)
print("Local diversity matrix:")
print(ld_df.round(3))

perm_stats = sld.compute_nd_permutation_stats(
    coords,
    labels,
    n_perm=100,
    radii=radii,
    random_state=7,
    n_jobs=1,
)
pval_df = perm_stats["pvals"]
perm_mean_df = perm_stats["perm_mean"]
perm_dist = perm_stats["distribution"]

print("\nPermutation p-values:")
print(pval_df.round(3))

print("\nPermutation null mean:")
print(perm_mean_df.round(3))
print(f"\nPermutation distribution shape: {perm_dist.shape}")

# Single-radius API
single_r = 1.1
single = sld.compute_local_diversity(coords, labels, radius=single_r)
print(f"\nSingle-radius local diversity (r={single_r}):")
print(np.round(single, 3))
