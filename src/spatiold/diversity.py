"""Core spatial local-diversity computations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree
from sklearn.neighbors import radius_neighbors_graph

DEFAULT_RADII: tuple[float, ...] = (20, 30, 40, 50, 75, 100, 150, 200, 250)
DEFAULT_GAUSSIAN_KERNEL_SUPPORT: float = 1.0
SpatialKernel = str | Callable[[np.ndarray], np.ndarray | Sequence[float] | float]


@dataclass(frozen=True)
class _NeighborGraph:
    """Compact CSR-style neighborhood graph for one radius."""

    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray | None = None


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


def _resolve_kernel(kernel: SpatialKernel) -> tuple[str, Callable[[np.ndarray], np.ndarray]]:
    if callable(kernel):
        def _callable_kernel(dist_over_r: np.ndarray) -> np.ndarray:
            raw = np.asarray(kernel(dist_over_r), dtype=float)
            if raw.ndim == 0:
                return np.full(dist_over_r.shape, float(raw), dtype=float)
            if raw.shape != dist_over_r.shape:
                raise ValueError(
                    "Callable `kernel` must return a scalar or an array with the same shape "
                    "as the input distances."
                )
            return raw

        return "callable", _callable_kernel

    name = str(kernel).strip().lower().replace("-", "_")
    if name in {"indicator", "binary", "uniform"}:
        return "indicator", lambda dist_over_r: (dist_over_r <= 1.0).astype(float)
    if name == "gaussian":
        return "gaussian", lambda dist_over_r: np.exp(-0.5 * np.square(dist_over_r))
    raise ValueError("`kernel` must be one of {'indicator', 'gaussian'} or a callable.")


def _kernel_requires_weights(kernel: SpatialKernel) -> bool:
    if callable(kernel):
        return True
    return str(kernel).strip().lower().replace("-", "_") not in {"indicator", "binary", "uniform"}


def _resolve_kernel_support(kernel_name: str, kernel_support: float | None) -> float | None:
    if kernel_support is not None:
        if np.isinf(kernel_support):
            return None
        if kernel_support <= 0:
            raise ValueError("`kernel_support` must be positive when provided.")

        if kernel_name == "indicator":
            if kernel_support < 1.0:
                raise ValueError("Indicator kernel requires `kernel_support >= 1`.")
            return float(kernel_support)
        return float(kernel_support)

    if kernel_name == "indicator":
        return 1.0
    if kernel_name == "gaussian":
        # Keep the weighted default local so runtime stays close to the legacy
        # fixed-radius neighborhood, while still allowing exact full-graph Gaussian
        # weights via `kernel_support=np.inf`.
        return DEFAULT_GAUSSIAN_KERNEL_SUPPORT
    return None


def _factorize_labels(labels: np.ndarray) -> tuple[np.ndarray, int]:
    label_codes, _ = pd.factorize(labels.astype(str), sort=False)
    label_codes = label_codes.astype(np.int64, copy=False)
    n_labels = int(label_codes.max()) + 1 if label_codes.size else 0
    return label_codes, n_labels


def _build_compact_neighbor_graph(
    coords_arr: np.ndarray,
    *,
    radius: float,
    include_self: bool,
    kernel_fn: Callable[[np.ndarray], np.ndarray] | None,
    kernel_support: float | None,
) -> _NeighborGraph:
    if kernel_support is None:
        n_cells = coords_arr.shape[0]
        all_idx = np.arange(n_cells, dtype=np.int32)
        counts = np.full(n_cells, n_cells, dtype=np.int64)
        if not include_self:
            counts -= 1

        indptr = np.empty(n_cells + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        indices = np.empty(int(indptr[-1]), dtype=np.int32)
        weights = np.empty(int(indptr[-1]), dtype=np.float32) if kernel_fn is not None else None

        for i in range(n_cells):
            start = int(indptr[i])
            end = int(indptr[i + 1])
            nbr_idx = all_idx if include_self else all_idx[all_idx != i]
            indices[start:end] = nbr_idx
            if weights is not None:
                scaled_dists = np.linalg.norm(coords_arr[nbr_idx] - coords_arr[i], axis=1) / float(radius)
                weight_arr = np.asarray(kernel_fn(scaled_dists), dtype=np.float32)
                if weight_arr.shape != nbr_idx.shape:
                    raise ValueError("Kernel weights must align with neighborhood indices.")
                weights[start:end] = weight_arr
        return _NeighborGraph(indptr=indptr, indices=indices, weights=weights)

    search_radius = float(radius) * float(kernel_support)
    mode = "distance" if kernel_fn is not None else "connectivity"
    graph = radius_neighbors_graph(
        coords_arr,
        radius=search_radius,
        mode=mode,
        include_self=include_self,
    ).tocsr()
    graph.sort_indices()

    if kernel_fn is None:
        return _NeighborGraph(
            indptr=graph.indptr.astype(np.int64, copy=False),
            indices=graph.indices.astype(np.int32, copy=False),
            weights=None,
        )

    weights = np.asarray(kernel_fn(graph.data / float(radius)), dtype=np.float32)
    if weights.shape != graph.data.shape:
        raise ValueError("Kernel weights must align with neighborhood indices.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("Kernel weights must be finite.")
    if np.any(weights < 0):
        raise ValueError("Kernel weights must be nonnegative.")

    graph = graph.copy()
    graph.data = weights
    graph.eliminate_zeros()
    graph.sort_indices()
    return _NeighborGraph(
        indptr=graph.indptr.astype(np.int64, copy=False),
        indices=graph.indices.astype(np.int32, copy=False),
        weights=graph.data.astype(np.float32, copy=False),
    )


def _indptr_from_edge_mask(base_indptr: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
    cumulative = np.empty(edge_mask.size + 1, dtype=np.int64)
    cumulative[0] = 0
    np.cumsum(edge_mask, dtype=np.int64, out=cumulative[1:])
    counts = cumulative[base_indptr[1:]] - cumulative[base_indptr[:-1]]

    indptr = np.empty(base_indptr.shape[0], dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return indptr


def _build_compact_neighbor_graphs_from_shared_distances(
    coords_arr: np.ndarray,
    radii: Sequence[float],
    *,
    include_self: bool,
    kernel_fn: Callable[[np.ndarray], np.ndarray] | None,
    kernel_support: float,
) -> list[_NeighborGraph]:
    max_search_radius = float(max(radii)) * float(kernel_support)
    base_graph = radius_neighbors_graph(
        coords_arr,
        radius=max_search_radius,
        mode="distance",
        include_self=include_self,
    ).tocsr()
    base_graph.sort_indices()

    base_indptr = base_graph.indptr.astype(np.int64, copy=False)
    base_indices = base_graph.indices.astype(np.int32, copy=False)
    base_distances = base_graph.data.astype(float, copy=False)

    graphs: list[_NeighborGraph] = []
    for radius in radii:
        search_radius = float(radius) * float(kernel_support)
        use_all_edges = search_radius >= max_search_radius
        distance_mask = None if use_all_edges else base_distances <= search_radius

        if kernel_fn is None:
            if use_all_edges:
                graphs.append(
                    _NeighborGraph(
                        indptr=base_indptr,
                        indices=base_indices,
                        weights=None,
                    )
                )
                continue
            if distance_mask is None:
                raise RuntimeError("Missing distance mask for radius-specific neighborhood graph.")
            graphs.append(
                _NeighborGraph(
                    indptr=_indptr_from_edge_mask(base_indptr, distance_mask),
                    indices=base_indices[distance_mask],
                    weights=None,
                )
            )
            continue

        selected_distances = base_distances if use_all_edges else base_distances[distance_mask]
        weights = np.asarray(kernel_fn(selected_distances / float(radius)), dtype=np.float32)
        if weights.shape != selected_distances.shape:
            raise ValueError("Kernel weights must align with neighborhood indices.")
        if not np.all(np.isfinite(weights)):
            raise ValueError("Kernel weights must be finite.")
        if np.any(weights < 0):
            raise ValueError("Kernel weights must be nonnegative.")

        positive = weights > 0
        if np.all(positive):
            if use_all_edges:
                graphs.append(
                    _NeighborGraph(
                        indptr=base_indptr,
                        indices=base_indices,
                        weights=weights.astype(np.float32, copy=False),
                    )
                )
                continue
            if distance_mask is None:
                raise RuntimeError("Missing distance mask for radius-specific neighborhood graph.")
            edge_mask = distance_mask
            graph_weights = weights
        else:
            selected_positions = (
                np.arange(base_distances.size, dtype=np.int64)
                if use_all_edges
                else np.flatnonzero(distance_mask)
            )
            edge_mask = np.zeros(base_distances.shape, dtype=bool)
            edge_mask[selected_positions[positive]] = True
            graph_weights = weights[positive]

        graphs.append(
            _NeighborGraph(
                indptr=_indptr_from_edge_mask(base_indptr, edge_mask),
                indices=base_indices[edge_mask],
                weights=graph_weights.astype(np.float32, copy=False),
            )
        )

    return graphs


def _precompute_neighbor_graphs(
    coords: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    radii: Sequence[float] | None = None,
    *,
    include_self: bool = True,
    kernel: SpatialKernel = "indicator",
    kernel_support: float | None = None,
) -> tuple[np.ndarray, list[float], list[_NeighborGraph]]:
    coords_arr = _as_2d_coords(coords)
    radii_list = _as_radii(radii)
    kernel_name, kernel_fn = _resolve_kernel(kernel)
    resolved_support = _resolve_kernel_support(kernel_name, kernel_support)
    use_weights = _kernel_requires_weights(kernel)

    radius_kernel_fn = kernel_fn if use_weights else None
    if resolved_support is not None and len(radii_list) > 1:
        graphs = _build_compact_neighbor_graphs_from_shared_distances(
            coords_arr,
            radii_list,
            include_self=include_self,
            kernel_fn=radius_kernel_fn,
            kernel_support=resolved_support,
        )
    else:
        graphs = [
            _build_compact_neighbor_graph(
                coords_arr,
                radius=radius,
                include_self=include_self,
                kernel_fn=radius_kernel_fn,
                kernel_support=resolved_support,
            )
            for radius in radii_list
        ]
    return coords_arr, radii_list, graphs


def _entropy_from_codes(
    label_codes: np.ndarray,
    n_labels: int,
    *,
    weights: np.ndarray | None = None,
    base: float = 2.0,
) -> float:
    if label_codes.size == 0:
        return 0.0

    if weights is None:
        probs = np.bincount(label_codes, minlength=n_labels).astype(float)
    else:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.shape != label_codes.shape:
            raise ValueError("Per-neighborhood weights must align with neighborhood indices.")
        probs = np.bincount(label_codes, weights=weights_arr, minlength=n_labels)

    total = probs.sum()
    if total <= 0:
        return 0.0

    probs = probs[probs > 0] / total
    log_probs = np.log(probs)
    if base is not None:
        if base <= 0 or base == 1:
            raise ValueError("`base` must be positive and not equal to 1.")
        log_probs = log_probs / np.log(base)

    entropy = float(-(probs * log_probs).sum())
    return 0.0 if abs(entropy) < 1e-12 else entropy


def _label_codes_to_sparse_indicator(label_codes: np.ndarray, n_labels: int) -> csr_matrix:
    n_cells = label_codes.shape[0]
    if n_cells == 0 or n_labels == 0:
        return csr_matrix((n_cells, n_labels), dtype=float)

    indicator = csr_matrix(
        (
            np.ones(n_cells, dtype=np.float32),
            (np.arange(n_cells, dtype=np.int64), label_codes),
        ),
        shape=(n_cells, n_labels),
    )
    return indicator


def _neighbor_graph_to_csr(graph: _NeighborGraph, n_cells: int) -> csr_matrix:
    if graph.indptr.shape[0] != n_cells + 1:
        raise ValueError("All radii in `neighbors_by_radius` must have the same number of cells.")
    data = (
        np.ones(graph.indices.shape[0], dtype=np.float32)
        if graph.weights is None
        else graph.weights
    )
    return csr_matrix((data, graph.indices, graph.indptr), shape=(n_cells, n_cells), copy=False)


def _entropy_from_sparse_label_weight_matrix(
    label_weights: csr_matrix,
    *,
    base: float = 2.0,
) -> np.ndarray:
    if base is not None and (base <= 0 or base == 1):
        raise ValueError("`base` must be positive and not equal to 1.")

    n_cells = label_weights.shape[0]
    entropy = np.zeros(n_cells, dtype=float)
    if label_weights.nnz == 0:
        return entropy

    totals = np.asarray(label_weights.sum(axis=1)).ravel().astype(float, copy=False)
    row_counts = np.diff(label_weights.indptr)
    rows = np.repeat(np.arange(n_cells, dtype=np.int64), row_counts)
    valid = totals[rows] > 0
    if not np.any(valid):
        return entropy

    probs = label_weights.data[valid].astype(float, copy=False) / totals[rows[valid]]
    positive = probs > 0
    terms = np.zeros_like(probs, dtype=float)
    terms[positive] = probs[positive] * np.log(probs[positive])
    if base is not None:
        terms[positive] = terms[positive] / np.log(base)

    entropy = -np.bincount(rows[valid], weights=terms, minlength=n_cells)
    entropy[np.abs(entropy) < 1e-12] = 0.0
    return entropy


def _compute_local_diversity_from_neighbor_graphs(
    label_codes: np.ndarray,
    n_labels: int,
    neighbors_by_radius: Sequence[_NeighborGraph],
    *,
    base: float = 2.0,
) -> np.ndarray:
    n_cells = int(label_codes.shape[0])
    label_indicator = _label_codes_to_sparse_indicator(label_codes, n_labels)
    output = np.zeros((len(neighbors_by_radius), n_cells), dtype=float)

    for ridx, graph in enumerate(neighbors_by_radius):
        graph_csr = _neighbor_graph_to_csr(graph, n_cells)
        label_weights = graph_csr @ label_indicator
        output[ridx] = _entropy_from_sparse_label_weight_matrix(label_weights, base=base)

    return output


def _finalize_neighbors_and_weights(
    nbr_idx: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[list[int], np.ndarray | None]:
    if weights is None:
        return nbr_idx.tolist(), None

    weights_arr = np.asarray(weights, dtype=float)
    if weights_arr.shape != nbr_idx.shape:
        raise ValueError("Kernel weights must align with neighborhood indices.")
    if not np.all(np.isfinite(weights_arr)):
        raise ValueError("Kernel weights must be finite.")
    if np.any(weights_arr < 0):
        raise ValueError("Kernel weights must be nonnegative.")

    positive = weights_arr > 0
    return nbr_idx[positive].tolist(), weights_arr[positive]


def _compute_local_diversity_from_label_codes(
    label_codes: np.ndarray,
    n_labels: int,
    neighbors_by_radius: Sequence[list[list[int]] | _NeighborGraph],
    *,
    weights_by_radius: list[list[np.ndarray]] | None = None,
    base: float = 2.0,
) -> np.ndarray:
    n_radii = len(neighbors_by_radius)
    if n_radii == 0:
        raise ValueError("`neighbors_by_radius` cannot be empty.")

    n_cells = int(label_codes.shape[0])
    if label_codes.shape[0] != n_cells:
        raise ValueError("`label_codes` length must match number of cells in `neighbors_by_radius`.")
    if weights_by_radius is not None and len(weights_by_radius) != n_radii:
        raise ValueError("`weights_by_radius` must align with `neighbors_by_radius`.")

    if all(isinstance(neighbors, _NeighborGraph) for neighbors in neighbors_by_radius):
        return _compute_local_diversity_from_neighbor_graphs(
            label_codes,
            n_labels,
            neighbors_by_radius,  # type: ignore[arg-type]
            base=base,
        )

    output = np.zeros((n_radii, n_cells), dtype=float)
    for ridx, neighbors in enumerate(neighbors_by_radius):
        if isinstance(neighbors, _NeighborGraph):
            if neighbors.indptr.shape[0] != n_cells + 1:
                raise ValueError("All radii in `neighbors_by_radius` must have the same number of cells.")

            for i in range(n_cells):
                start = int(neighbors.indptr[i])
                end = int(neighbors.indptr[i + 1])
                if start == end:
                    continue

                nbr_codes = label_codes[neighbors.indices[start:end]]
                nbr_weights = None if neighbors.weights is None else neighbors.weights[start:end]
                output[ridx, i] = _entropy_from_codes(
                    nbr_codes,
                    n_labels,
                    weights=nbr_weights,
                    base=base,
                )
            continue

        if len(neighbors) != n_cells:
            raise ValueError("All radii in `neighbors_by_radius` must have the same number of cells.")
        radius_weights = None if weights_by_radius is None else weights_by_radius[ridx]
        if radius_weights is not None and len(radius_weights) != n_cells:
            raise ValueError("`weights_by_radius` must align with `neighbors_by_radius`.")

        for i, nbr_idx in enumerate(neighbors):
            if len(nbr_idx) == 0:
                continue

            nbr_codes = label_codes[nbr_idx]
            nbr_weights = None if radius_weights is None else radius_weights[i]
            output[ridx, i] = _entropy_from_codes(
                nbr_codes,
                n_labels,
                weights=nbr_weights,
                base=base,
            )

    return output


def _precompute_neighbors_for_radius(
    coords_arr: np.ndarray,
    tree: KDTree,
    radius: float,
    *,
    include_self: bool,
    kernel_fn: Callable[[np.ndarray], np.ndarray] | None,
    kernel_support: float | None,
) -> tuple[list[list[int]], list[np.ndarray] | None]:
    n_cells = coords_arr.shape[0]
    neighbors: list[list[int]] = []
    weights_by_cell: list[np.ndarray] | None = [] if kernel_fn is not None else None

    if kernel_support is None:
        all_idx = np.arange(n_cells, dtype=np.int64)
        for i in range(n_cells):
            nbr_idx = all_idx if include_self else all_idx[all_idx != i]
            scaled_dists = np.linalg.norm(coords_arr[nbr_idx] - coords_arr[i], axis=1) / float(radius)
            nbr_list, weight_arr = _finalize_neighbors_and_weights(nbr_idx, kernel_fn(scaled_dists))
            neighbors.append(nbr_list)
            if weights_by_cell is not None and weight_arr is not None:
                weights_by_cell.append(weight_arr)
        return neighbors, weights_by_cell

    search_radius = float(radius) * float(kernel_support)
    raw_neighbors = tree.query_ball_point(coords_arr, r=search_radius)
    for i, nbrs in enumerate(raw_neighbors):
        nbr_idx = np.asarray(nbrs, dtype=np.int64)
        if not include_self:
            nbr_idx = nbr_idx[nbr_idx != i]

        if kernel_fn is None:
            neighbors.append(nbr_idx.tolist())
            continue

        scaled_dists = np.linalg.norm(coords_arr[nbr_idx] - coords_arr[i], axis=1) / float(radius)
        nbr_list, weight_arr = _finalize_neighbors_and_weights(nbr_idx, kernel_fn(scaled_dists))
        neighbors.append(nbr_list)
        if weights_by_cell is not None and weight_arr is not None:
            weights_by_cell.append(weight_arr)

    return neighbors, weights_by_cell


def precompute_neighbors(
    coords: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    radii: Sequence[float] | None = None,
    *,
    include_self: bool = True,
    kernel: SpatialKernel = "indicator",
    kernel_support: float | None = None,
    return_weights: bool = False,
) -> (
    tuple[np.ndarray, list[float], list[list[list[int]]]]
    | tuple[np.ndarray, list[float], list[list[list[int]]], list[list[np.ndarray]]]
):
    """Precompute neighborhood indices, and optionally spatial weights, for each radius.

    Returns ``(coords_arr, radii_list, neighbors_by_radius)`` by default. When
    ``return_weights=True``, a fourth output ``weights_by_radius`` is returned,
    where ``weights_by_radius[r_idx][i]`` aligns with
    ``neighbors_by_radius[r_idx][i]``.

    Notes
    -----
    - ``kernel="indicator"`` reproduces the legacy fixed-radius neighborhood.
    - ``kernel="gaussian"`` uses weights ``exp(-0.5 * (d / r)^2)``.
    - Built-in kernels default to local support so runtime stays close to the
      legacy fixed-radius pipeline. For Gaussian, set ``kernel_support=np.inf``
      for exact full-graph evaluation, or use a finite support multiplier to
      truncate evaluation to neighbors within ``kernel_support * radius``.
    """
    coords_arr = _as_2d_coords(coords)
    radii_list = _as_radii(radii)
    kernel_name, kernel_fn = _resolve_kernel(kernel)
    resolved_support = _resolve_kernel_support(kernel_name, kernel_support)

    tree = KDTree(coords_arr)
    neighbors_by_radius: list[list[list[int]]] = []
    weights_by_radius: list[list[np.ndarray]] = []

    for radius in radii_list:
        use_weights = return_weights or kernel_name != "indicator"
        radius_kernel_fn = kernel_fn if use_weights else None
        neighbors, weights = _precompute_neighbors_for_radius(
            coords_arr,
            tree,
            radius,
            include_self=include_self,
            kernel_fn=radius_kernel_fn,
            kernel_support=resolved_support,
        )
        neighbors_by_radius.append(neighbors)
        if return_weights:
            if weights is None:
                weights = [np.ones(len(nbr_idx), dtype=float) for nbr_idx in neighbors]
            weights_by_radius.append(weights)

    if return_weights:
        return coords_arr, radii_list, neighbors_by_radius, weights_by_radius
    return coords_arr, radii_list, neighbors_by_radius


def compute_local_diversity_from_neighbors(
    labels: pd.Series | np.ndarray | Sequence[object],
    neighbors_by_radius: Sequence[list[list[int]] | _NeighborGraph],
    *,
    weights_by_radius: list[list[np.ndarray]] | None = None,
    base: float = 2.0,
) -> np.ndarray:
    """Compute local diversity for precomputed neighborhoods.

    Parameters
    ----------
    labels
        Cell labels.
    neighbors_by_radius
        Neighborhood index lists for each radius.
    weights_by_radius
        Optional per-edge spatial weights aligned with ``neighbors_by_radius``.
        When omitted, all neighbors contribute equally.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_radii, n_cells)``.
    """
    labels_arr = np.asarray(labels)
    n_radii = len(neighbors_by_radius)
    if n_radii == 0:
        raise ValueError("`neighbors_by_radius` cannot be empty.")

    first_radius = neighbors_by_radius[0]
    if isinstance(first_radius, _NeighborGraph):
        n_cells = int(first_radius.indptr.shape[0] - 1)
    else:
        n_cells = len(first_radius)
    if labels_arr.shape[0] != n_cells:
        raise ValueError("`labels` length must match number of cells in `neighbors_by_radius`.")
    if weights_by_radius is not None and len(weights_by_radius) != n_radii:
        raise ValueError("`weights_by_radius` must align with `neighbors_by_radius`.")

    label_codes, n_labels = _factorize_labels(labels_arr)
    return _compute_local_diversity_from_label_codes(
        label_codes,
        n_labels,
        neighbors_by_radius,
        weights_by_radius=weights_by_radius,
        base=base,
    )


def compute_local_diversity(
    coords: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    labels: pd.Series | np.ndarray | Sequence[object],
    *,
    radius: float,
    include_self: bool = True,
    base: float = 2.0,
    kernel: SpatialKernel = "indicator",
    kernel_support: float | None = None,
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
    kernel
        Spatial weighting kernel. ``"indicator"`` reproduces the legacy fixed-radius
        score; ``"gaussian"`` computes a distance-weighted local composition before
        Shannon entropy.
    kernel_support
        Optional support multiplier for kernel evaluation. By default, Gaussian
        weights are truncated at ``1 * radius`` to keep runtime close to the
        legacy pipeline. Use ``kernel_support=np.inf`` for exact full-graph
        Gaussian weights.
    """
    coords_arr = _as_2d_coords(coords)
    labels_arr = _as_labels(labels, coords_arr.shape[0])

    _, _, neighbor_graphs = _precompute_neighbor_graphs(
        coords_arr,
        [radius],
        include_self=include_self,
        kernel=kernel,
        kernel_support=kernel_support,
    )
    result = compute_local_diversity_from_neighbors(
        labels_arr,
        neighbor_graphs,
        base=base,
    )
    return result[0]


def compute_local_diversity_multi_radius(
    coords: pd.DataFrame | np.ndarray | Sequence[Sequence[float]],
    labels: pd.Series | np.ndarray | Sequence[object],
    *,
    radii: Sequence[float] | None = None,
    include_self: bool = True,
    base: float = 2.0,
    kernel: SpatialKernel = "indicator",
    kernel_support: float | None = None,
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

    _, radii_list, neighbor_graphs = _precompute_neighbor_graphs(
        coords_arr,
        radii=radii,
        include_self=include_self,
        kernel=kernel,
        kernel_support=kernel_support,
    )
    matrix = compute_local_diversity_from_neighbors(
        labels_arr,
        neighbor_graphs,
        base=base,
    )
    return pd.DataFrame(matrix.T, index=cell_ids, columns=radii_list)


# Backward-compatible alias used in prototype notebooks.
compute_neighborhood_diversity = compute_local_diversity
