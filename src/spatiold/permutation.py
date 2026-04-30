"""Permutation testing utilities for spatial local diversity."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .diversity import (
    SpatialKernel,
    _compute_local_diversity_from_label_codes,
    _precompute_neighbor_graphs,
    _factorize_labels,
)

_NEIGHBORS_G: list[object] | None = None
_WEIGHTS_G: list[list[np.ndarray]] | None = None
_LABEL_CODES_G: np.ndarray | None = None
_N_LABELS_G: int | None = None
_BASE_G: float = 2.0
_PVAL_POOLING_ALIASES: dict[str, str] = {
    "cell": "cell",
    "self": "cell",
    "global": "global",
    "all": "global",
    "neighborhood_size": "neighborhood_size",
    "neighborhood-size": "neighborhood_size",
    "neighbor_count": "neighborhood_size",
    "neighbor-count": "neighborhood_size",
    "matched": "neighborhood_size",
}
_ZSCORE_EPSILON_DEFAULT: float = 1e-8


def _init_perm_worker(
    neighbors_by_radius: list[object],
    weights_by_radius: list[list[np.ndarray]] | None,
    label_codes: np.ndarray,
    n_labels: int,
    base: float,
) -> None:
    global _NEIGHBORS_G, _WEIGHTS_G, _LABEL_CODES_G, _N_LABELS_G, _BASE_G
    _NEIGHBORS_G = neighbors_by_radius
    _WEIGHTS_G = weights_by_radius
    _LABEL_CODES_G = label_codes
    _N_LABELS_G = n_labels
    _BASE_G = base


def _perm_worker(seed: int) -> np.ndarray:
    if _NEIGHBORS_G is None or _LABEL_CODES_G is None or _N_LABELS_G is None:
        raise RuntimeError("Permutation worker was not initialized.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(_LABEL_CODES_G)
    return _compute_local_diversity_from_label_codes(
        shuffled,
        _N_LABELS_G,
        _NEIGHBORS_G,
        weights_by_radius=_WEIGHTS_G,
        base=_BASE_G,
    )


def _run_permutations(
    label_codes: np.ndarray,
    n_labels: int,
    neighbors_by_radius: list[list[list[int]]],
    weights_by_radius: list[list[np.ndarray]] | None,
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
            shuffled = np.random.default_rng(int(seed)).permutation(label_codes)
            yield _compute_local_diversity_from_label_codes(
                shuffled,
                n_labels,
                neighbors_by_radius,
                weights_by_radius=weights_by_radius,
                base=base,
            )
        return

    ctx = mp.get_context("fork" if os.name != "nt" else "spawn")
    chunksize = max(1, n_perm // (n_jobs * 4))
    with ctx.Pool(
        processes=n_jobs,
        initializer=_init_perm_worker,
        initargs=(neighbors_by_radius, weights_by_radius, label_codes, n_labels, base),
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


def _resolve_pval_pooling(pval_pooling: str) -> str:
    mode = pval_pooling.strip().lower().replace(" ", "_")
    try:
        return _PVAL_POOLING_ALIASES[mode]
    except KeyError as exc:
        raise ValueError(
            "`pval_pooling` must be one of {'cell', 'global', 'neighborhood_size'}."
        ) from exc


def _build_neighbor_count_groups(
    neighbors_by_radius: list[object],
) -> list[dict[int, np.ndarray]]:
    groups_by_radius: list[dict[int, np.ndarray]] = []
    for neighbors in neighbors_by_radius:
        if hasattr(neighbors, "indptr"):
            counts = np.diff(getattr(neighbors, "indptr")).astype(np.int64, copy=False)
        else:
            counts = np.fromiter((len(nbr_idx) for nbr_idx in neighbors), dtype=np.int64, count=len(neighbors))
        groups = {
            int(count): np.flatnonzero(counts == count)
            for count in np.unique(counts)
        }
        groups_by_radius.append(groups)
    return groups_by_radius


def _init_group_moment_accumulators(
    groups_by_radius: list[dict[int, np.ndarray]],
    *,
    track_sumsq: bool,
) -> tuple[list[dict[int, float]], list[dict[int, float]] | None]:
    group_sums = [{count: 0.0 for count in groups} for groups in groups_by_radius]
    group_sumsq = (
        [{count: 0.0 for count in groups} for groups in groups_by_radius]
        if track_sumsq
        else None
    )
    return group_sums, group_sumsq


def _update_group_moments(
    perm_matrix: np.ndarray,
    groups_by_radius: list[dict[int, np.ndarray]],
    group_sums: list[dict[int, float]],
    group_sumsq: list[dict[int, float]] | None,
) -> None:
    for ridx, groups in enumerate(groups_by_radius):
        perm_row = perm_matrix[ridx]
        for count, idx in groups.items():
            vals = perm_row[idx]
            group_sums[ridx][count] += float(vals.sum())
            if group_sumsq is not None:
                group_sumsq[ridx][count] += float(np.square(vals).sum())


def _finalize_group_moments(
    *,
    groups_by_radius: list[dict[int, np.ndarray]],
    group_sums: list[dict[int, float]],
    group_sumsq: list[dict[int, float]] | None,
    n_perm: int,
    n_radii: int,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    mean = np.zeros((n_radii, n_cells), dtype=float)
    std = np.zeros((n_radii, n_cells), dtype=float) if group_sumsq is not None else None

    for ridx, groups in enumerate(groups_by_radius):
        for count, idx in groups.items():
            pooled_n = int(n_perm * idx.size)
            if pooled_n <= 0:
                continue

            pooled_mean = group_sums[ridx][count] / float(pooled_n)
            mean[ridx, idx] = pooled_mean

            if std is None or group_sumsq is None:
                continue
            if pooled_n <= 1:
                std[ridx, idx] = 0.0
                continue

            pooled_var = (group_sumsq[ridx][count] - pooled_n * (pooled_mean**2)) / float(pooled_n - 1)
            std[ridx, idx] = np.sqrt(max(float(pooled_var), 0.0))

    return mean, std


def _build_pval_denominator(
    *,
    n_perm: int,
    n_radii: int,
    n_cells: int,
    pval_pooling: str,
    groups_by_radius: list[dict[int, np.ndarray]] | None,
) -> np.ndarray:
    if pval_pooling == "cell":
        return np.full((n_radii, n_cells), n_perm, dtype=np.int64)
    if pval_pooling == "global":
        return np.full((n_radii, n_cells), n_perm * n_cells, dtype=np.int64)
    if groups_by_radius is None:
        raise RuntimeError("Missing neighborhood-size groups for pooled p-value computation.")

    denominator = np.zeros((n_radii, n_cells), dtype=np.int64)
    for ridx, groups in enumerate(groups_by_radius):
        for idx in groups.values():
            denominator[ridx, idx] = n_perm * idx.size
    return denominator


def _update_cellwise_extremes(
    *,
    greater_counts: np.ndarray,
    less_counts: np.ndarray,
    observed: np.ndarray,
    perm_matrix: np.ndarray,
) -> None:
    greater_counts += perm_matrix >= observed
    less_counts += perm_matrix <= observed


def _update_global_pooled_extremes(
    *,
    greater_counts: np.ndarray,
    less_counts: np.ndarray,
    observed: np.ndarray,
    perm_matrix: np.ndarray,
) -> None:
    for ridx in range(perm_matrix.shape[0]):
        perm_sorted = np.sort(perm_matrix[ridx])
        greater_counts[ridx] += perm_sorted.size - np.searchsorted(
            perm_sorted,
            observed[ridx],
            side="left",
        )
        less_counts[ridx] += np.searchsorted(
            perm_sorted,
            observed[ridx],
            side="right",
        )


def _update_neighbor_count_pooled_extremes(
    *,
    greater_counts: np.ndarray,
    less_counts: np.ndarray,
    observed: np.ndarray,
    perm_matrix: np.ndarray,
    groups_by_radius: list[dict[int, np.ndarray]],
) -> None:
    for ridx, groups in enumerate(groups_by_radius):
        perm_row = perm_matrix[ridx]
        obs_row = observed[ridx]
        for idx in groups.values():
            perm_sorted = np.sort(perm_row[idx])
            greater_counts[ridx, idx] += perm_sorted.size - np.searchsorted(
                perm_sorted,
                obs_row[idx],
                side="left",
            )
            less_counts[ridx, idx] += np.searchsorted(
                perm_sorted,
                obs_row[idx],
                side="right",
            )


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
    kernel: SpatialKernel = "indicator",
    kernel_support: float | None = None,
    pval_pooling: str = "neighborhood_size",
    need_pvals: bool,
    need_mean: bool,
    need_std: bool,
    need_distribution: bool,
    need_permutation_means: bool,
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    if n_perm < 1:
        raise ValueError("`n_perm` must be >= 1.")

    n_jobs_eff = _resolve_jobs(n_jobs)
    coords_arr, radii_list, neighbors_by_radius = _precompute_neighbor_graphs(
        xy,
        radii=radii,
        include_self=include_self,
        kernel=kernel,
        kernel_support=kernel_support,
    )
    weights_by_radius = None
    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != coords_arr.shape[0]:
        raise ValueError("`xy` and `labels` must have the same number of rows.")
    label_codes, n_labels = _factorize_labels(labels_arr)

    n_radii = len(radii_list)
    n_cells = coords_arr.shape[0]
    cell_ids = _resolve_cell_ids(xy, labels, n_cells=n_cells)
    groups_by_radius = _build_neighbor_count_groups(neighbors_by_radius)

    observed = None
    greater_counts = None
    less_counts = None
    pval_denominator = None
    if need_pvals:
        observed = _compute_local_diversity_from_label_codes(
            label_codes,
            n_labels,
            neighbors_by_radius,
            weights_by_radius=weights_by_radius,
            base=base,
        )
        pval_pooling_mode = _resolve_pval_pooling(pval_pooling)
        greater_counts = np.zeros_like(observed, dtype=np.int64)
        less_counts = np.zeros_like(observed, dtype=np.int64)
        pval_denominator = _build_pval_denominator(
            n_perm=n_perm,
            n_radii=n_radii,
            n_cells=n_cells,
            pval_pooling=pval_pooling_mode,
            groups_by_radius=groups_by_radius,
        )
    else:
        pval_pooling_mode = "cell"

    track_group_moments = need_mean or need_std
    group_sums = None
    group_sumsq = None
    if track_group_moments:
        group_sums, group_sumsq = _init_group_moment_accumulators(
            groups_by_radius,
            track_sumsq=need_std or need_pvals,
        )

    perm_dist = np.empty((n_perm, n_radii, n_cells), dtype=float) if need_distribution else None
    perm_means = np.empty((n_perm, n_radii), dtype=float) if need_permutation_means else None

    for perm_idx, perm_matrix in enumerate(
        _run_permutations(
            label_codes,
            n_labels,
            neighbors_by_radius,
            weights_by_radius,
            n_perm=n_perm,
            random_state=random_state,
            n_jobs=n_jobs_eff,
            base=base,
        )
    ):
        if perm_dist is not None:
            perm_dist[perm_idx] = perm_matrix
        if perm_means is not None:
            perm_means[perm_idx] = perm_matrix.mean(axis=1)
        if group_sums is not None:
            _update_group_moments(
                perm_matrix,
                groups_by_radius,
                group_sums,
                group_sumsq,
            )
        if observed is not None and greater_counts is not None and less_counts is not None:
            if pval_pooling_mode == "cell":
                _update_cellwise_extremes(
                    greater_counts=greater_counts,
                    less_counts=less_counts,
                    observed=observed,
                    perm_matrix=perm_matrix,
                )
            elif pval_pooling_mode == "global":
                _update_global_pooled_extremes(
                    greater_counts=greater_counts,
                    less_counts=less_counts,
                    observed=observed,
                    perm_matrix=perm_matrix,
                )
            else:
                if groups_by_radius is None:
                    raise RuntimeError("Missing neighborhood-size groups for pooled p-values.")
                _update_neighbor_count_pooled_extremes(
                    greater_counts=greater_counts,
                    less_counts=less_counts,
                    observed=observed,
                    perm_matrix=perm_matrix,
                    groups_by_radius=groups_by_radius,
                )

    pvals_mixing_df = None
    pvals_segregation_df = None
    pvals_two_sided_df = None
    if (
        observed is not None
        and greater_counts is not None
        and less_counts is not None
        and pval_denominator is not None
    ):
        p_greater = (greater_counts + 1) / (pval_denominator + 1)
        p_less = (less_counts + 1) / (pval_denominator + 1)
        pvals_mixing_df = pd.DataFrame(p_greater.T, columns=radii_list, index=cell_ids)
        pvals_segregation_df = pd.DataFrame(p_less.T, columns=radii_list, index=cell_ids)
        pvals_two_sided_df = pd.DataFrame(
            np.minimum(1.0, 2.0 * np.minimum(p_greater, p_less)).T,
            columns=radii_list,
            index=cell_ids,
        )

    perm_mean_df = None
    perm_std_df = None
    zscore_df = None
    if group_sums is not None:
        mean_matrix, std_matrix = _finalize_group_moments(
            groups_by_radius=groups_by_radius,
            group_sums=group_sums,
            group_sumsq=group_sumsq if (need_std or need_pvals) else None,
            n_perm=n_perm,
            n_radii=n_radii,
            n_cells=n_cells,
        )
        if need_mean:
            perm_mean_df = pd.DataFrame(mean_matrix.T, columns=radii_list, index=cell_ids)
        if need_std:
            if std_matrix is None:
                raise RuntimeError("Missing permutation-null standard deviation matrix.")
            perm_std_df = pd.DataFrame(std_matrix.T, columns=radii_list, index=cell_ids)
        if need_pvals:
            if observed is None or std_matrix is None:
                raise RuntimeError("Missing observed values or null standard deviation for z-score computation.")
            zscore = (observed - mean_matrix) / (std_matrix + _ZSCORE_EPSILON_DEFAULT)
            zscore_df = pd.DataFrame(zscore.T, columns=radii_list, index=cell_ids)

    return (
        pvals_mixing_df,
        pvals_segregation_df,
        pvals_two_sided_df,
        perm_mean_df,
        perm_std_df,
        zscore_df,
        perm_dist,
        perm_means,
    )


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
    kernel: SpatialKernel = "indicator",
    kernel_support: float | None = None,
    pval_pooling: str = "neighborhood_size",
    return_distribution: bool = True,
    return_permutation_means: bool = False,
) -> dict[str, pd.DataFrame | np.ndarray]:
    """Compute permutation p-values and matched null moments in one pass.

    Parameters
    ----------
    pval_pooling
        Strategy for pooling permutation-null draws when computing cell-level
        p-values. ``"cell"`` reproduces the legacy per-cell comparison,
        ``"global"`` pools across all permuted cells at a radius, and
        ``"neighborhood_size"`` pools across permuted cells with the same
        neighborhood size at that radius. This is the default.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``"pvals_mixing"``: one-sided upper-tail DataFrame ``(n_cells, n_radii)``
        - ``"pvals_segregation"``: one-sided lower-tail DataFrame ``(n_cells, n_radii)``
        - ``"pvals_two_sided"``: two-sided DataFrame ``(n_cells, n_radii)``
        - ``"perm_mean"``: matched-null mean DataFrame ``(n_cells, n_radii)``
        - ``"perm_std"``: matched-null standard deviation DataFrame ``(n_cells, n_radii)``
        - ``"zscore"``: null-standardized observed score DataFrame ``(n_cells, n_radii)``
        - ``"distribution"``: ndarray ``(n_perm, n_radii, n_cells)`` when
          ``return_distribution=True``
        - ``"permutation_means"``: ndarray ``(n_perm, n_radii)`` when
          ``return_permutation_means=True``
    """
    (
        pvals_mixing_df,
        pvals_segregation_df,
        pvals_two_sided_df,
        perm_mean_df,
        perm_std_df,
        zscore_df,
        perm_dist,
        perm_means,
    ) = _compute_nd_permutation_outputs(
        xy,
        labels,
        n_perm,
        radii=radii,
        random_state=random_state,
        n_jobs=n_jobs,
        include_self=include_self,
        base=base,
        kernel=kernel,
        kernel_support=kernel_support,
        pval_pooling=pval_pooling,
        need_pvals=True,
        need_mean=True,
        need_std=True,
        need_distribution=return_distribution,
        need_permutation_means=return_permutation_means,
    )
    if (
        pvals_mixing_df is None
        or pvals_segregation_df is None
        or pvals_two_sided_df is None
        or perm_mean_df is None
        or perm_std_df is None
        or zscore_df is None
        or (return_distribution and perm_dist is None)
        or (return_permutation_means and perm_means is None)
    ):
        raise RuntimeError("Failed to compute permutation outputs.")
    stats: dict[str, pd.DataFrame | np.ndarray] = {
        "pvals_mixing": pvals_mixing_df,
        "pvals_segregation": pvals_segregation_df,
        "pvals_two_sided": pvals_two_sided_df,
        "perm_mean": perm_mean_df,
        "perm_std": perm_std_df,
        "zscore": zscore_df,
        "distribution": perm_dist,
    }
    if return_permutation_means:
        if perm_means is None:
            raise RuntimeError("Missing per-permutation radius means.")
        stats["permutation_means"] = perm_means
    return stats


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
    kernel: SpatialKernel = "indicator",
    kernel_support: float | None = None,
) -> pd.DataFrame:
    """Compute matched permutation null mean for neighborhood diversity.

    Returns a DataFrame with shape ``(n_cells, n_radii)``.
    """
    _, _, _, perm_mean_df, _, _, _, _ = _compute_nd_permutation_outputs(
        xy,
        labels,
        n_perm,
        radii=radii,
        random_state=random_state,
        n_jobs=n_jobs,
        include_self=include_self,
        base=base,
        kernel=kernel,
        kernel_support=kernel_support,
        need_pvals=False,
        need_mean=True,
        need_std=False,
        need_distribution=False,
        need_permutation_means=False,
    )
    if perm_mean_df is None:
        raise RuntimeError("Failed to compute permutation mean.")
    return perm_mean_df


def compute_nd_permutation_std(
    xy: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    labels: pd.Series | np.ndarray | Sequence[object],
    n_perm: int,
    *,
    radii: Sequence[float] | None = None,
    random_state: int = 42,
    n_jobs: int | None = None,
    include_self: bool = True,
    base: float = 2.0,
    kernel: SpatialKernel = "indicator",
    kernel_support: float | None = None,
) -> pd.DataFrame:
    """Compute matched permutation null standard deviation for neighborhood diversity."""
    _, _, _, _, perm_std_df, _, _, _ = _compute_nd_permutation_outputs(
        xy,
        labels,
        n_perm,
        radii=radii,
        random_state=random_state,
        n_jobs=n_jobs,
        include_self=include_self,
        base=base,
        kernel=kernel,
        kernel_support=kernel_support,
        need_pvals=False,
        need_mean=False,
        need_std=True,
        need_distribution=False,
        need_permutation_means=False,
    )
    if perm_std_df is None:
        raise RuntimeError("Failed to compute permutation standard deviation.")
    return perm_std_df


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
    kernel: SpatialKernel = "indicator",
    kernel_support: float | None = None,
) -> np.ndarray:
    """Return full permutation diversity distribution.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_perm, n_radii, n_cells)``.
    """
    _, _, _, _, _, _, perm_dist, _ = _compute_nd_permutation_outputs(
        xy,
        labels,
        n_perm,
        radii=radii,
        random_state=random_state,
        n_jobs=n_jobs,
        include_self=include_self,
        base=base,
        kernel=kernel,
        kernel_support=kernel_support,
        need_pvals=False,
        need_mean=False,
        need_std=False,
        need_distribution=True,
        need_permutation_means=False,
    )
    if perm_dist is None:
        raise RuntimeError("Failed to compute permutation distribution.")
    return perm_dist
