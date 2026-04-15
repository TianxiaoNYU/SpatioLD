from __future__ import annotations

import numpy as np
import pandas as pd

from spatiold import (
    compute_local_diversity_multi_radius,
    compute_nd_permutation_distribution,
    compute_nd_permutation_mean,
    compute_nd_permutation_pvals,
    compute_nd_permutation_stats,
)
from spatiold.diversity import precompute_neighbors


def test_permutation_pvals_shape_range_and_reproducibility() -> None:
    coords = pd.DataFrame({"x": [0, 0, 1, 1, 2], "y": [0, 1, 0, 1, 0]})
    labels = pd.Series(["A", "A", "B", "B", "B"], index=[f"c{i}" for i in range(5)])

    p1 = compute_nd_permutation_pvals(
        coords,
        labels,
        n_perm=25,
        radii=[0.25, 2.0],
        random_state=123,
        n_jobs=1,
    )
    p2 = compute_nd_permutation_pvals(
        coords,
        labels,
        n_perm=25,
        radii=[0.25, 2.0],
        random_state=123,
        n_jobs=1,
    )

    assert p1.shape == (5, 2)
    assert np.all((p1.values >= 0) & (p1.values <= 1))
    assert p1.equals(p2)


def test_neighborhood_size_pooling_matches_manual_distribution_pooling() -> None:
    coords = pd.DataFrame({"x": [0.0, 1.0, 2.5, 5.5], "y": [0.0, 0.0, 0.0, 0.0]})
    labels = pd.Series(["A", "A", "B", "B"], index=[f"c{i}" for i in range(4)])
    radius = 2.0

    stats = compute_nd_permutation_stats(
        coords,
        labels,
        n_perm=9,
        radii=[radius],
        random_state=3,
        n_jobs=1,
        pval_pooling="neighborhood_size",
    )

    _, _, neighbors_by_radius = precompute_neighbors(coords, radii=[radius])
    group_keys = np.array([len(nbr_idx) for nbr_idx in neighbors_by_radius[0]])
    observed = compute_local_diversity_multi_radius(coords, labels, radii=[radius]).iloc[:, 0]
    dist = stats["distribution"][:, 0, :]

    expected = np.empty(len(labels), dtype=float)
    for count in np.unique(group_keys):
        idx = np.flatnonzero(group_keys == count)
        pooled = dist[:, idx].reshape(-1)
        expected[idx] = ((pooled[:, None] >= observed.iloc[idx].to_numpy()[None, :]).sum(axis=0) + 1) / (
            pooled.size + 1
        )

    assert np.allclose(stats["pvals"].iloc[:, 0].to_numpy(), expected)


def test_global_pooling_matches_manual_distribution_pooling() -> None:
    coords = pd.DataFrame({"x": [0.0, 1.0, 2.5, 5.5], "y": [0.0, 0.0, 0.0, 0.0]})
    labels = pd.Series(["A", "A", "B", "B"], index=[f"c{i}" for i in range(4)])
    radius = 2.0

    stats = compute_nd_permutation_stats(
        coords,
        labels,
        n_perm=7,
        radii=[radius],
        random_state=5,
        n_jobs=1,
        pval_pooling="global",
    )

    observed = compute_local_diversity_multi_radius(coords, labels, radii=[radius]).iloc[:, 0]
    pooled = stats["distribution"][:, 0, :].reshape(-1)
    expected = ((pooled[:, None] >= observed.to_numpy()[None, :]).sum(axis=0) + 1) / (pooled.size + 1)

    assert np.allclose(stats["pvals"].iloc[:, 0].to_numpy(), expected)


def test_permutation_mean_shape() -> None:
    coords = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 0, 1]})
    labels = pd.Series(["A", "A", "B", "B"])

    mean_df = compute_nd_permutation_mean(
        coords,
        labels,
        n_perm=20,
        radii=[0.5, 2.0],
        random_state=11,
        n_jobs=1,
    )

    assert mean_df.shape == (4, 2)
    assert (mean_df.values >= 0).all()


def test_permutation_distribution_shape() -> None:
    coords = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 0, 1]})
    labels = pd.Series(["A", "A", "B", "B"])

    dist = compute_nd_permutation_distribution(
        coords,
        labels,
        n_perm=12,
        radii=[0.5, 2.0],
        random_state=5,
        n_jobs=1,
    )

    assert dist.shape == (12, 2, 4)
    assert np.isfinite(dist).all()


def test_combined_permutation_stats_shape() -> None:
    coords = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 0, 1]})
    labels = pd.Series(["A", "A", "B", "B"], index=[f"c{i}" for i in range(4)])

    stats = compute_nd_permutation_stats(
        coords,
        labels,
        n_perm=10,
        radii=[0.5, 2.0],
        random_state=5,
        n_jobs=1,
    )

    assert set(stats.keys()) == {"pvals", "perm_mean", "distribution"}
    assert stats["pvals"].shape == (4, 2)
    assert stats["perm_mean"].shape == (4, 2)
    assert stats["distribution"].shape == (10, 2, 4)
