from __future__ import annotations

import numpy as np
import pandas as pd

from spatiold import (
    compute_local_diversity_multi_radius,
    compute_nd_permutation_distribution,
    compute_nd_permutation_mean,
    compute_nd_permutation_stats,
    compute_nd_permutation_std,
)
from spatiold.diversity import precompute_neighbors


def test_permutation_stats_pvals_shape_range_and_reproducibility() -> None:
    coords = pd.DataFrame({"x": [0, 0, 1, 1, 2], "y": [0, 1, 0, 1, 0]})
    labels = pd.Series(["A", "A", "B", "B", "B"], index=[f"c{i}" for i in range(5)])

    stats1 = compute_nd_permutation_stats(
        coords,
        labels,
        n_perm=25,
        radii=[0.25, 2.0],
        random_state=123,
        n_jobs=1,
    )
    stats2 = compute_nd_permutation_stats(
        coords,
        labels,
        n_perm=25,
        radii=[0.25, 2.0],
        random_state=123,
        n_jobs=1,
    )

    for key in ["pvals_mixing", "pvals_segregation", "pvals_two_sided"]:
        assert stats1[key].shape == (5, 2)
        assert np.all((stats1[key].values >= 0) & (stats1[key].values <= 1))
        assert stats1[key].equals(stats2[key])


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

    expected_mixing = np.empty(len(labels), dtype=float)
    expected_segregation = np.empty(len(labels), dtype=float)
    for count in np.unique(group_keys):
        idx = np.flatnonzero(group_keys == count)
        pooled = dist[:, idx].reshape(-1)
        expected_mixing[idx] = ((pooled[:, None] >= observed.iloc[idx].to_numpy()[None, :]).sum(axis=0) + 1) / (
            pooled.size + 1
        )
        expected_segregation[idx] = ((pooled[:, None] <= observed.iloc[idx].to_numpy()[None, :]).sum(axis=0) + 1) / (
            pooled.size + 1
        )

    expected_two_sided = np.minimum(1.0, 2.0 * np.minimum(expected_mixing, expected_segregation))

    assert np.allclose(stats["pvals_mixing"].iloc[:, 0].to_numpy(), expected_mixing)
    assert np.allclose(stats["pvals_segregation"].iloc[:, 0].to_numpy(), expected_segregation)
    assert np.allclose(stats["pvals_two_sided"].iloc[:, 0].to_numpy(), expected_two_sided)


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
    expected_mixing = ((pooled[:, None] >= observed.to_numpy()[None, :]).sum(axis=0) + 1) / (pooled.size + 1)
    expected_segregation = ((pooled[:, None] <= observed.to_numpy()[None, :]).sum(axis=0) + 1) / (pooled.size + 1)

    assert np.allclose(stats["pvals_mixing"].iloc[:, 0].to_numpy(), expected_mixing)
    assert np.allclose(stats["pvals_segregation"].iloc[:, 0].to_numpy(), expected_segregation)
    assert np.allclose(
        stats["pvals_two_sided"].iloc[:, 0].to_numpy(),
        np.minimum(1.0, 2.0 * np.minimum(expected_mixing, expected_segregation)),
    )


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

    assert set(stats.keys()) == {
        "pvals_mixing",
        "pvals_segregation",
        "pvals_two_sided",
        "perm_mean",
        "perm_std",
        "zscore",
        "distribution",
    }
    assert stats["pvals_mixing"].shape == (4, 2)
    assert stats["pvals_segregation"].shape == (4, 2)
    assert stats["pvals_two_sided"].shape == (4, 2)
    assert stats["perm_mean"].shape == (4, 2)
    assert stats["perm_std"].shape == (4, 2)
    assert stats["zscore"].shape == (4, 2)
    assert stats["distribution"].shape == (10, 2, 4)


def test_combined_permutation_stats_can_skip_full_distribution() -> None:
    coords = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 0, 1]})
    labels = pd.Series(["A", "A", "B", "B"], index=[f"c{i}" for i in range(4)])

    stats = compute_nd_permutation_stats(
        coords,
        labels,
        n_perm=10,
        radii=[0.5, 2.0],
        random_state=5,
        n_jobs=1,
        return_distribution=False,
        return_permutation_means=True,
    )
    dist = compute_nd_permutation_distribution(
        coords,
        labels,
        n_perm=10,
        radii=[0.5, 2.0],
        random_state=5,
        n_jobs=1,
    )

    assert stats["distribution"] is None
    assert stats["permutation_means"].shape == (10, 2)
    assert np.allclose(stats["permutation_means"], dist.mean(axis=2))


def test_weighted_kernel_neighborhood_size_pooling_matches_manual_distribution_pooling() -> None:
    coords = pd.DataFrame({"x": [0.0, 0.5, 1.5, 3.0], "y": [0.0, 0.0, 0.0, 0.0]})
    labels = pd.Series(["A", "A", "B", "B"], index=[f"c{i}" for i in range(4)])
    radius = 1.0

    stats = compute_nd_permutation_stats(
        coords,
        labels,
        n_perm=8,
        radii=[radius],
        random_state=7,
        n_jobs=1,
        include_self=False,
        kernel="gaussian",
        pval_pooling="neighborhood_size",
    )

    _, _, neighbors_by_radius = precompute_neighbors(
        coords,
        radii=[radius],
        include_self=False,
        kernel="gaussian",
        kernel_support=1.0,
    )
    group_keys = np.array([len(nbr_idx) for nbr_idx in neighbors_by_radius[0]])
    observed = compute_local_diversity_multi_radius(
        coords,
        labels,
        radii=[radius],
        include_self=False,
        kernel="gaussian",
    ).iloc[:, 0]
    dist = stats["distribution"][:, 0, :]

    expected_mixing = np.empty(len(labels), dtype=float)
    expected_segregation = np.empty(len(labels), dtype=float)
    for count in np.unique(group_keys):
        idx = np.flatnonzero(group_keys == count)
        pooled = dist[:, idx].reshape(-1)
        expected_mixing[idx] = ((pooled[:, None] >= observed.iloc[idx].to_numpy()[None, :]).sum(axis=0) + 1) / (
            pooled.size + 1
        )
        expected_segregation[idx] = ((pooled[:, None] <= observed.iloc[idx].to_numpy()[None, :]).sum(axis=0) + 1) / (
            pooled.size + 1
        )

    assert np.allclose(stats["pvals_mixing"].iloc[:, 0].to_numpy(), expected_mixing)
    assert np.allclose(stats["pvals_segregation"].iloc[:, 0].to_numpy(), expected_segregation)
    assert np.allclose(
        stats["pvals_two_sided"].iloc[:, 0].to_numpy(),
        np.minimum(1.0, 2.0 * np.minimum(expected_mixing, expected_segregation)),
    )


def test_matched_null_mean_and_std_match_manual_group_pooling() -> None:
    coords = pd.DataFrame({"x": [0.0, 1.0, 2.5, 5.5], "y": [0.0, 0.0, 0.0, 0.0]})
    labels = pd.Series(["A", "A", "B", "B"], index=[f"c{i}" for i in range(4)])
    radius = 2.0

    mean_df = compute_nd_permutation_mean(
        coords,
        labels,
        n_perm=9,
        radii=[radius],
        random_state=3,
        n_jobs=1,
    )
    std_df = compute_nd_permutation_std(
        coords,
        labels,
        n_perm=9,
        radii=[radius],
        random_state=3,
        n_jobs=1,
    )
    dist = compute_nd_permutation_distribution(
        coords,
        labels,
        n_perm=9,
        radii=[radius],
        random_state=3,
        n_jobs=1,
    )[:, 0, :]

    _, _, neighbors_by_radius = precompute_neighbors(coords, radii=[radius])
    group_keys = np.array([len(nbr_idx) for nbr_idx in neighbors_by_radius[0]])

    expected_mean = np.empty(len(labels), dtype=float)
    expected_std = np.empty(len(labels), dtype=float)
    for count in np.unique(group_keys):
        idx = np.flatnonzero(group_keys == count)
        pooled = dist[:, idx].reshape(-1)
        expected_mean[idx] = pooled.mean()
        expected_std[idx] = pooled.std(ddof=1) if pooled.size > 1 else 0.0

    assert np.allclose(mean_df.iloc[:, 0].to_numpy(), expected_mean)
    assert np.allclose(std_df.iloc[:, 0].to_numpy(), expected_std)
