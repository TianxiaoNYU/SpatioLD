from __future__ import annotations

import numpy as np
import pandas as pd

from spatiold import (
    compute_nd_permutation_distribution,
    compute_nd_permutation_mean,
    compute_nd_permutation_pvals,
    compute_nd_permutation_stats,
)


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
