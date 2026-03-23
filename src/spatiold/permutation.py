"""Permutation testing utilities for spatial local diversity."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .diversity import (
    compute_local_diversity_from_neighbors,
    precompute_neighbors,
)

_NEIGHBORS_G: list[list[list[int]]] | None = None
_LABELS_G: np.ndarray | None = None
_BASE_G: float = 2.0


def _init_perm_worker(
    neighbors_by_radius: list[list[list[int]]], labels: np.ndarray, base: float
) -> None:
    global _NEIGHBORS_G, _LABELS_G, _BASE_G
    _NEIGHBORS_G = neighbors_by_radius
    _LABELS_G = labels
    _BASE_G = base


def _perm_worker(seed: int) -> np.ndarray:
    if _NEIGHBORS_G is None or _LABELS_G is None:
        raise RuntimeError("Permutation worker was not initialized.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(_LABELS_G)
    return compute_local_diversity_from_neighbors(shuffled, _NEIGHBORS_G, base=_BASE_G)


def _run_permutations(
    labels_arr: np.ndarray,
    neighbors_by_radius: list[list[list[int]]],
    *,
    n_perm: int,
    random_state: int,
    n_jobs: int,
    base: float,
):
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, np.iinfo(np.uint32).max, size=n_perm, dtype=np.uint32)

    if n_jobs == 1:
        for seed in seeds:
            shuffled = np.random.default_rng(int(seed)).permutation(labels_arr)
            yield compute_local_diversity_from_neighbors(
                shuffled, neighbors_by_radius, base=base
            )
        return

    ctx = mp.get_context("fork" if os.name != "nt" else "spawn")
    chunksize = max(1, n_perm // (n_jobs * 4))
    with ctx.Pool(
        processes=n_jobs,
        initializer=_init_perm_worker,
        initargs=(neighbors_by_radius, labels_arr, base),
    ) as pool:
        for perm_matrix in pool.imap(_perm_worker, seeds.tolist(), chunksize=chunksize):
            yield perm_matrix


def _resolve_jobs(n_jobs: int | None) -> int:
    if n_jobs is None:
        return max(1, mp.cpu_count() - 1)
    if n_jobs == -1:
        return max(1, mp.cpu_count() - 1)
    if n_jobs < 1:
        raise ValueError("`n_jobs` must be >= 1, -1, or None.")
    return n_jobs


def compute_nd_permutation_pvals(
    xy: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    labels: pd.Series | np.ndarray | Sequence[object],
    n_perm: int,
    *,
    radii: Sequence[float] | None = None,
    random_state: int = 42,
    n_jobs: int | None = None,
    include_self: bool = True,
    base: float = 2.0,
    alternative: str = "greater",
) -> pd.DataFrame:
    """Compute permutation p-values for neighborhood diversity.

    Parameters
    ----------
    xy
        Spatial coordinates of cells.
    labels
        Cell labels (for example cell type).
    n_perm
        Number of label permutations.
    alternative
        One of ``{"greater", "less", "two-sided"}``.
    """
    if n_perm < 1:
        raise ValueError("`n_perm` must be >= 1.")

    n_jobs_eff = _resolve_jobs(n_jobs)

    coords_arr, radii_list, neighbors_by_radius = precompute_neighbors(
        xy, radii=radii, include_self=include_self
    )
    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != coords_arr.shape[0]:
        raise ValueError("`xy` and `labels` must have the same number of rows.")

    observed = compute_local_diversity_from_neighbors(
        labels_arr, neighbors_by_radius, base=base
    )

    alt = alternative.lower()
    if alt not in {"greater", "less", "two-sided"}:
        raise ValueError("`alternative` must be one of {'greater', 'less', 'two-sided' }.")

    greater_counts = np.zeros_like(observed, dtype=np.int64)
    less_counts = np.zeros_like(observed, dtype=np.int64)

    for perm_matrix in _run_permutations(
        labels_arr,
        neighbors_by_radius,
        n_perm=n_perm,
        random_state=random_state,
        n_jobs=n_jobs_eff,
        base=base,
    ):
        greater_counts += perm_matrix >= observed
        less_counts += perm_matrix <= observed

    p_greater = (greater_counts + 1) / (n_perm + 1)
    p_less = (less_counts + 1) / (n_perm + 1)

    if alt == "greater":
        pvals = p_greater
    elif alt == "less":
        pvals = p_less
    else:
        pvals = np.minimum(1.0, 2.0 * np.minimum(p_greater, p_less))

    if hasattr(labels, "index"):
        cell_ids = getattr(labels, "index").astype(str)
    elif hasattr(xy, "index"):
        cell_ids = getattr(xy, "index").astype(str)
    else:
        cell_ids = np.arange(coords_arr.shape[0]).astype(str)

    return pd.DataFrame(pvals.T, columns=radii_list, index=cell_ids)


def compute_nd_permutation_mean(
    xy: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    labels: pd.Series | np.ndarray | Sequence[object],
    n_perm: int,
    *,
    radii: Sequence[float] | None = None,
    random_state: int = 42,
    n_jobs: int | None = None,
    include_self: bool = True,
    base: float = 2.0,
) -> pd.DataFrame:
    """Compute permutation null mean for neighborhood diversity.

    Returns a DataFrame with shape ``(n_cells, n_radii)``.
    """
    if n_perm < 1:
        raise ValueError("`n_perm` must be >= 1.")

    n_jobs_eff = _resolve_jobs(n_jobs)

    coords_arr, radii_list, neighbors_by_radius = precompute_neighbors(
        xy, radii=radii, include_self=include_self
    )
    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != coords_arr.shape[0]:
        raise ValueError("`xy` and `labels` must have the same number of rows.")

    perm_sum = np.zeros((len(radii_list), coords_arr.shape[0]), dtype=float)
    for perm_matrix in _run_permutations(
        labels_arr,
        neighbors_by_radius,
        n_perm=n_perm,
        random_state=random_state,
        n_jobs=n_jobs_eff,
        base=base,
    ):
        perm_sum += perm_matrix

    perm_mean = perm_sum / n_perm

    if hasattr(labels, "index"):
        cell_ids = getattr(labels, "index").astype(str)
    elif hasattr(xy, "index"):
        cell_ids = getattr(xy, "index").astype(str)
    else:
        cell_ids = np.arange(coords_arr.shape[0]).astype(str)

    return pd.DataFrame(perm_mean.T, columns=radii_list, index=cell_ids)


def compute_nd_permutation_distribution(
    xy: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    labels: pd.Series | np.ndarray | Sequence[object],
    n_perm: int,
    *,
    radii: Sequence[float] | None = None,
    random_state: int = 42,
    n_jobs: int | None = None,
    include_self: bool = True,
    base: float = 2.0,
) -> np.ndarray:
    """Return full permutation diversity distribution.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_perm, n_radii, n_cells)``.
    """
    if n_perm < 1:
        raise ValueError("`n_perm` must be >= 1.")

    n_jobs_eff = _resolve_jobs(n_jobs)
    coords_arr, _, neighbors_by_radius = precompute_neighbors(
        xy, radii=radii, include_self=include_self
    )
    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != coords_arr.shape[0]:
        raise ValueError("`xy` and `labels` must have the same number of rows.")

    perms = list(
        _run_permutations(
            labels_arr,
            neighbors_by_radius,
            n_perm=n_perm,
            random_state=random_state,
            n_jobs=n_jobs_eff,
            base=base,
        )
    )
    return np.asarray(perms, dtype=float)
