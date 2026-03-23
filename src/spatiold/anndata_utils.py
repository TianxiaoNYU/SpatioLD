"""AnnData helpers for SpatioLD."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def extract_coords_from_anndata(
    adata,
    *,
    coord_keys: Sequence[str] | None = None,
    spatial_key: str = "spatial",
) -> np.ndarray:
    """Extract coordinates from ``adata.obs`` or ``adata.obsm``.

    Priority is:
    1. ``coord_keys`` in ``adata.obs`` (for example ``("x", "y")``)
    2. ``adata.obsm[spatial_key]`` first two columns
    """
    if coord_keys is not None:
        if len(coord_keys) < 2:
            raise ValueError("`coord_keys` must contain at least two column names.")
        if all(k in adata.obs.columns for k in coord_keys[:2]):
            return adata.obs.loc[:, list(coord_keys[:2])].to_numpy(dtype=float)

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"Unable to find coordinates. Neither coord_keys={coord_keys} in adata.obs "
            f"nor adata.obsm['{spatial_key}'] is available."
        )

    spatial = np.asarray(adata.obsm[spatial_key])
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError(
            f"adata.obsm['{spatial_key}'] must have shape (n_cells, >=2). Got {spatial.shape}."
        )
    return spatial[:, :2].astype(float, copy=False)


def extract_labels_from_anndata(adata, *, label_key: str) -> pd.Series:
    """Extract cell labels from ``adata.obs[label_key]``."""
    if label_key not in adata.obs.columns:
        raise KeyError(f"`label_key='{label_key}'` not found in adata.obs columns.")

    labels = adata.obs[label_key].copy()
    labels.index = adata.obs_names.astype(str)
    return labels


def store_matrix_in_anndata(adata, matrix_df: pd.DataFrame, *, key: str) -> None:
    """Store a cell x radius matrix in ``adata.obsm`` with radius metadata in ``adata.uns``."""
    if matrix_df.shape[0] != adata.n_obs:
        raise ValueError("matrix row count must equal adata.n_obs.")

    matrix_aligned = matrix_df.reindex(adata.obs_names.astype(str))
    adata.obsm[key] = matrix_aligned.to_numpy(dtype=float)
    adata.uns[f"{key}_radii"] = [float(c) for c in matrix_aligned.columns]


def obsm_matrix_to_df(adata, *, key: str) -> pd.DataFrame:
    """Convert an ``obsm`` matrix back to a DataFrame with recovered radius columns."""
    if key not in adata.obsm:
        raise KeyError(f"`{key}` not found in adata.obsm.")

    matrix = np.asarray(adata.obsm[key])
    if matrix.ndim != 2:
        raise ValueError(f"adata.obsm['{key}'] must be 2D. Got shape {matrix.shape}.")

    cols = adata.uns.get(f"{key}_radii", list(range(matrix.shape[1])))
    return pd.DataFrame(matrix, index=adata.obs_names.astype(str), columns=cols)
