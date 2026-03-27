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


def _resolve_cell_ids(
    xy: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    labels: pd.Series | np.ndarray | Sequence[object],
    n_cells: int,
) -> pd.Index:
    if hasattr(labels, "index"):
        return pd.Index(getattr(labels, "index").astype(str))
    if hasattr(xy, "index"):
        return pd.Index(getattr(xy, "index").astype(str))
    return pd.Index(np.arange(n_cells).astype(str))


def _compute_nd_permutation_outputs(
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
    need_pvals: bool,
    need_mean: bool,
    need_distribution: bool,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, np.ndarray | None]:
    if n_perm < 1:
        raise ValueError("`n_perm` must be >= 1.")

    n_jobs_eff = _resolve_jobs(n_jobs)
    coords_arr, radii_list, neighbors_by_radius = precompute_neighbors(
        xy, radii=radii, include_self=include_self
    )
    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != coords_arr.shape[0]:
        raise ValueError("`xy` and `labels` must have the same number of rows.")

    n_radii = len(radii_list)
    n_cells = coords_arr.shape[0]
    cell_ids = _resolve_cell_ids(xy, labels, n_cells=n_cells)

    observed = None
    greater_counts = None
    less_counts = None
    if need_pvals:
        observed = compute_local_diversity_from_neighbors(
            labels_arr, neighbors_by_radius, base=base
        )
        alt = alternative.lower()
        if alt not in {"greater", "less", "two-sided"}:
            raise ValueError("`alternative` must be one of {'greater', 'less', 'two-sided' }.")
        greater_counts = np.zeros_like(observed, dtype=np.int64)
        less_counts = np.zeros_like(observed, dtype=np.int64)
    else:
        alt = "greater"

    perm_sum = np.zeros((n_radii, n_cells), dtype=float) if need_mean else None
    perm_dist = np.empty((n_perm, n_radii, n_cells), dtype=float) if need_distribution else None

    for perm_idx, perm_matrix in enumerate(
        _run_permutations(
            labels_arr,
            neighbors_by_radius,
            n_perm=n_perm,
            random_state=random_state,
            n_jobs=n_jobs_eff,
            base=base,
        )
    ):
        if perm_dist is not None:
            perm_dist[perm_idx] = perm_matrix
        if perm_sum is not None:
            perm_sum += perm_matrix
        if observed is not None and greater_counts is not None and less_counts is not None:
            greater_counts += perm_matrix >= observed
            less_counts += perm_matrix <= observed

    pvals_df = None
    if observed is not None and greater_counts is not None and less_counts is not None:
        p_greater = (greater_counts + 1) / (n_perm + 1)
        p_less = (less_counts + 1) / (n_perm + 1)

        if alt == "greater":
            pvals = p_greater
        elif alt == "less":
            pvals = p_less
        else:
            pvals = np.minimum(1.0, 2.0 * np.minimum(p_greater, p_less))
        pvals_df = pd.DataFrame(pvals.T, columns=radii_list, index=cell_ids)

    perm_mean_df = None
    if perm_sum is not None:
        perm_mean_df = pd.DataFrame((perm_sum / n_perm).T, columns=radii_list, index=cell_ids)

    return pvals_df, perm_mean_df, perm_dist


def compute_nd_permutation_stats(
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
) -> dict[str, pd.DataFrame | np.ndarray]:
    """Compute permutation p-values, null mean, and distribution in one pass.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``"pvals"``: DataFrame ``(n_cells, n_radii)``
        - ``"perm_mean"``: DataFrame ``(n_cells, n_radii)``
        - ``"distribution"``: ndarray ``(n_perm, n_radii, n_cells)``
    """
    pvals_df, perm_mean_df, perm_dist = _compute_nd_permutation_outputs(
        xy,
        labels,
        n_perm,
        radii=radii,
        random_state=random_state,
        n_jobs=n_jobs,
        include_self=include_self,
        base=base,
        alternative=alternative,
        need_pvals=True,
        need_mean=True,
        need_distribution=True,
    )
    if pvals_df is None or perm_mean_df is None or perm_dist is None:
        raise RuntimeError("Failed to compute full permutation outputs.")
    return {
        "pvals": pvals_df,
        "perm_mean": perm_mean_df,
        "distribution": perm_dist,
    }


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
    pvals_df, _, _ = _compute_nd_permutation_outputs(
        xy,
        labels,
        n_perm,
        radii=radii,
        random_state=random_state,
        n_jobs=n_jobs,
        include_self=include_self,
        base=base,
        alternative=alternative,
        need_pvals=True,
        need_mean=False,
        need_distribution=False,
    )
    if pvals_df is None:
        raise RuntimeError("Failed to compute permutation p-values.")
    return pvals_df


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
    _, perm_mean_df, _ = _compute_nd_permutation_outputs(
        xy,
        labels,
        n_perm,
        radii=radii,
        random_state=random_state,
        n_jobs=n_jobs,
        include_self=include_self,
        base=base,
        need_pvals=False,
        need_mean=True,
        need_distribution=False,
    )
    if perm_mean_df is None:
        raise RuntimeError("Failed to compute permutation mean.")
    return perm_mean_df


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
    _, _, perm_dist = _compute_nd_permutation_outputs(
        xy,
        labels,
        n_perm,
        radii=radii,
        random_state=random_state,
        n_jobs=n_jobs,
        include_self=include_self,
        base=base,
        need_pvals=False,
        need_mean=False,
        need_distribution=True,
    )
    if perm_dist is None:
        raise RuntimeError("Failed to compute permutation distribution.")
    return perm_dist
