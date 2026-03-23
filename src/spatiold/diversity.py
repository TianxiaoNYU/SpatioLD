"""Core spatial local-diversity computations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

DEFAULT_RADII: tuple[float, ...] = (20, 30, 40, 50, 75, 100, 150, 200, 250)


def _as_2d_coords(coords: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    coords_arr = np.asarray(coords)
    if coords_arr.ndim != 2 or coords_arr.shape[1] < 2:
        raise ValueError(f"`coords` must have shape (n_cells, >=2). Got {coords_arr.shape}.")
    return coords_arr[:, :2].astype(float, copy=False)


def _as_labels(labels: pd.Series | np.ndarray | Sequence[object], n_cells: int) -> np.ndarray:
    labels_arr = np.asarray(labels)
    if labels_arr.shape[0] != n_cells:
        raise ValueError(
            f"`labels` length ({labels_arr.shape[0]}) does not match number of coordinates ({n_cells})."
        )
    return labels_arr


def _as_radii(radii: Sequence[float] | None) -> list[float]:
    radii_list = list(DEFAULT_RADII if radii is None else radii)
    if not radii_list:
        raise ValueError("`radii` must contain at least one radius.")
    if any(r <= 0 for r in radii_list):
        raise ValueError("All `radii` values must be positive.")
    return [float(r) for r in radii_list]


def _cell_ids(
    coords: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    labels: pd.Series | np.ndarray | Sequence[object],
    n_cells: int,
) -> pd.Index:
    if hasattr(labels, "index"):
        return pd.Index(getattr(labels, "index").astype(str))
    if hasattr(coords, "index"):
        return pd.Index(getattr(coords, "index").astype(str))
    return pd.Index(np.arange(n_cells).astype(str))


def _entropy_from_labels(neighbor_labels: np.ndarray, base: float = 2.0) -> float:
    if neighbor_labels.size == 0:
        return 0.0

    _, counts = np.unique(neighbor_labels, return_counts=True)
    probs = counts / counts.sum()
    log_probs = np.log(probs)
    if base is not None:
        if base <= 0 or base == 1:
            raise ValueError("`base` must be positive and not equal to 1.")
        log_probs = log_probs / np.log(base)

    return float(-(probs * log_probs).sum())


def precompute_neighbors(
    coords: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    radii: Sequence[float] | None = None,
    *,
    include_self: bool = True,
) -> tuple[np.ndarray, list[float], list[list[list[int]]]]:
    """Precompute neighbor index lists for each radius.

    Returns a tuple: ``(coords_arr, radii_list, neighbors_by_radius)`` where
    ``neighbors_by_radius[r_idx][i]`` is the list of neighbor indices for cell ``i``.
    """
    coords_arr = _as_2d_coords(coords)
    radii_list = _as_radii(radii)

    tree = KDTree(coords_arr)
    neighbors_by_radius: list[list[list[int]]] = []

    for radius in radii_list:
        indices = tree.query_ball_point(coords_arr, r=radius)
        if not include_self:
            indices = [[j for j in nbrs if j != i] for i, nbrs in enumerate(indices)]
        neighbors_by_radius.append(indices)

    return coords_arr, radii_list, neighbors_by_radius


def compute_local_diversity_from_neighbors(
    labels: pd.Series | np.ndarray | Sequence[object],
    neighbors_by_radius: list[list[list[int]]],
    *,
    base: float = 2.0,
) -> np.ndarray:
    """Compute local diversity for precomputed neighborhoods.

    Returns an array of shape ``(n_radii, n_cells)``.
    """
    labels_arr = np.asarray(labels)
    n_radii = len(neighbors_by_radius)
    if n_radii == 0:
        raise ValueError("`neighbors_by_radius` cannot be empty.")

    n_cells = len(neighbors_by_radius[0])
    if labels_arr.shape[0] != n_cells:
        raise ValueError("`labels` length must match number of cells in `neighbors_by_radius`.")

    output = np.zeros((n_radii, n_cells), dtype=float)
    for ridx, neighbors in enumerate(neighbors_by_radius):
        for i, nbr_idx in enumerate(neighbors):
            if not nbr_idx:
                continue
            neighbor_labels = labels_arr[nbr_idx]
            output[ridx, i] = _entropy_from_labels(neighbor_labels, base=base)

    return output


def compute_local_diversity(
    coords: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    labels: pd.Series | np.ndarray | Sequence[object],
    *,
    radius: float,
    include_self: bool = True,
    base: float = 2.0,
) -> np.ndarray:
    """Compute per-cell local diversity at a single radius.

    Parameters
    ----------
    coords
        Spatial coordinates of cells with shape ``(n_cells, 2)`` (or larger; first
        two columns are used).
    labels
        Categorical label for each cell (for example cell type).
    radius
        Neighborhood radius in the same unit as ``coords``.
    include_self
        Whether each cell is included in its own neighborhood.
    base
        Log base in Shannon entropy. Default is 2.
    """
    coords_arr = _as_2d_coords(coords)
    labels_arr = _as_labels(labels, coords_arr.shape[0])

    _, _, neighbors_by_radius = precompute_neighbors(
        coords_arr, [radius], include_self=include_self
    )
    result = compute_local_diversity_from_neighbors(labels_arr, neighbors_by_radius, base=base)
    return result[0]


def compute_local_diversity_multi_radius(
    coords: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    labels: pd.Series | np.ndarray | Sequence[object],
    *,
    radii: Sequence[float] | None = None,
    include_self: bool = True,
    base: float = 2.0,
) -> pd.DataFrame:
    """Compute per-cell local diversity across multiple radii.

    Returns
    -------
    pd.DataFrame
        DataFrame with shape ``(n_cells, n_radii)`` indexed by cell IDs.
    """
    coords_arr = _as_2d_coords(coords)
    labels_arr = _as_labels(labels, coords_arr.shape[0])
    cell_ids = _cell_ids(coords, labels, coords_arr.shape[0])

    _, radii_list, neighbors_by_radius = precompute_neighbors(
        coords_arr, radii=radii, include_self=include_self
    )
    matrix = compute_local_diversity_from_neighbors(labels_arr, neighbors_by_radius, base=base)
    return pd.DataFrame(matrix.T, index=cell_ids, columns=radii_list)


# Backward-compatible alias used in prototype notebooks.
compute_neighborhood_diversity = compute_local_diversity
