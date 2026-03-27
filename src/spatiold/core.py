"""High-level SpatioLD object API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .anndata_utils import (
    extract_coords_from_anndata,
    extract_labels_from_anndata,
    obsm_matrix_to_df,
    store_matrix_in_anndata,
)
from .diversity import DEFAULT_RADII, compute_local_diversity_multi_radius
from .permutation import (
    compute_nd_permutation_distribution,
    compute_nd_permutation_mean,
    compute_nd_permutation_pvals,
    compute_nd_permutation_stats,
)
from .pipeline import (
    build_significance_mask as _build_significance_mask,
    cluster_local_diversity_profiles as _cluster_local_diversity_profiles,
    compute_global_shannon_entropy as _compute_global_shannon_entropy,
    compute_sample_vs_null_summary as _compute_sample_vs_null_summary,
    compute_svg_morans_i as _compute_svg_morans_i,
    fit_slide_level_cell_type_radius_model as _fit_slide_level_cell_type_radius_model,
    prepare_shared_components as _prepare_shared_components,
    summarize_model_terms as _summarize_model_terms,
    summarize_slide_level_cell_type_effects as _summarize_slide_level_cell_type_effects,
    summarize_local_diversity_by_cell_type as _summarize_local_diversity_by_cell_type,
)


@dataclass
class SpatioLD:
    """AnnData-first interface for local diversity analysis.

    Parameters
    ----------
    adata
        Input AnnData object.
    label_key
        Column in ``adata.obs`` containing cell labels (for example cell type).
    coord_keys
        Optional coordinate columns in ``adata.obs``. If omitted, coordinates are
        read from ``adata.obsm[spatial_key]``.
    spatial_key
        Key in ``adata.obsm`` used when ``coord_keys`` is not provided.
    """

    adata: Any
    label_key: str = "cell_type"
    coord_keys: tuple[str, str] | None = None
    spatial_key: str = "spatial"

    def __post_init__(self) -> None:
        extract_labels_from_anndata(self.adata, label_key=self.label_key)
        extract_coords_from_anndata(
            self.adata,
            coord_keys=self.coord_keys,
            spatial_key=self.spatial_key,
        )

    @classmethod
    def from_anndata(
        cls,
        adata,
        *,
        label_key: str = "cell_type",
        coord_keys: tuple[str, str] | None = None,
        spatial_key: str = "spatial",
        copy: bool = False,
    ) -> "SpatioLD":
        """Create a ``SpatioLD`` object from an existing AnnData object."""
        adata_obj = adata.copy() if copy else adata
        return cls(
            adata=adata_obj,
            label_key=label_key,
            coord_keys=coord_keys,
            spatial_key=spatial_key,
        )

    @classmethod
    def from_arrays(
        cls,
        coords: pd.DataFrame | np.ndarray,
        labels: pd.Series | np.ndarray,
        *,
        label_key: str = "cell_type",
        cell_ids: list[str] | None = None,
    ) -> "SpatioLD":
        """Create an AnnData-backed ``SpatioLD`` object from arrays."""
        import anndata as ad

        coords_arr = np.asarray(coords)
        labels_arr = np.asarray(labels)

        if coords_arr.ndim != 2 or coords_arr.shape[1] < 2:
            raise ValueError(f"`coords` must have shape (n_cells, >=2). Got {coords_arr.shape}.")
        if labels_arr.shape[0] != coords_arr.shape[0]:
            raise ValueError("`labels` length must match number of coordinate rows.")

        n_cells = coords_arr.shape[0]
        if cell_ids is None:
            cell_ids = [str(i) for i in range(n_cells)]
        if len(cell_ids) != n_cells:
            raise ValueError("`cell_ids` length must match number of coordinate rows.")

        obs = pd.DataFrame({label_key: labels_arr}, index=pd.Index(cell_ids, name="cell_id"))
        adata = ad.AnnData(X=np.empty((n_cells, 0)), obs=obs)
        adata.obsm["spatial"] = coords_arr[:, :2].astype(float, copy=False)

        return cls(adata=adata, label_key=label_key, spatial_key="spatial")

    @property
    def coords(self) -> np.ndarray:
        """Cell coordinates as ``(n_cells, 2)`` array."""
        return extract_coords_from_anndata(
            self.adata,
            coord_keys=self.coord_keys,
            spatial_key=self.spatial_key,
        )

    @property
    def labels(self) -> pd.Series:
        """Cell labels as a Series indexed by ``obs_names``."""
        return extract_labels_from_anndata(self.adata, label_key=self.label_key)

    @property
    def metadata(self) -> pd.DataFrame:
        """Cell metadata (`adata.obs`) indexed by `obs_names`."""
        meta = self.adata.obs.copy()
        meta.index = self.adata.obs_names.astype(str)
        return meta

    def to_anndata(self):
        """Return the backing AnnData object."""
        return self.adata

    def compute_global_shannon_entropy(
        self,
        *,
        base: float = 2.0,
    ) -> float:
        """Compute global Shannon entropy from object-held labels."""
        return _compute_global_shannon_entropy(self.labels, base=base)

    def get_coords_df(
        self,
        *,
        x_col: str = "x",
        y_col: str = "y",
    ) -> pd.DataFrame:
        """Return coordinates as a DataFrame indexed by cell ID.

        If `x_col`/`y_col` exist in metadata, those columns are returned.
        Otherwise, coordinates are derived from `self.coords`.
        """
        meta = self.metadata
        if x_col in meta.columns and y_col in meta.columns:
            return meta[[x_col, y_col]].copy()

        coords = self.coords
        return pd.DataFrame(coords, index=meta.index, columns=[x_col, y_col])

    def get_result(self, key: str) -> pd.DataFrame:
        """Read a stored result matrix from ``adata.obsm``."""
        return obsm_matrix_to_df(self.adata, key=key)

    def compute_local_diversity(
        self,
        *,
        radii: list[float] | list[int] | tuple[float, ...] = DEFAULT_RADII,
        include_self: bool = True,
        base: float = 2.0,
        store: bool = True,
        key: str = "spatiold_local_diversity",
    ) -> pd.DataFrame:
        """Compute local diversity matrix (cells x radii)."""
        ld_df = compute_local_diversity_multi_radius(
            self.coords,
            self.labels,
            radii=radii,
            include_self=include_self,
            base=base,
        )
        if store:
            store_matrix_in_anndata(self.adata, ld_df, key=key)
        return ld_df

    def compute_permutation_pvals(
        self,
        n_perm: int,
        *,
        radii: list[float] | tuple[float, ...] = DEFAULT_RADII,
        random_state: int = 42,
        n_jobs: int | None = None,
        include_self: bool = True,
        base: float = 2.0,
        alternative: str = "greater",
        store: bool = True,
        key: str = "spatiold_local_diversity_pvals",
    ) -> pd.DataFrame:
        """Compute permutation p-values for local diversity."""
        pval_df = compute_nd_permutation_pvals(
            self.coords,
            self.labels,
            n_perm=n_perm,
            radii=radii,
            random_state=random_state,
            n_jobs=n_jobs,
            include_self=include_self,
            base=base,
            alternative=alternative,
        )
        if store:
            store_matrix_in_anndata(self.adata, pval_df, key=key)
        return pval_df

    def compute_permutation_stats(
        self,
        n_perm: int,
        *,
        radii: list[float] | tuple[float, ...] = DEFAULT_RADII,
        random_state: int = 42,
        n_jobs: int | None = None,
        include_self: bool = True,
        base: float = 2.0,
        alternative: str = "greater",
        store: bool = True,
        pvals_key: str = "spatiold_local_diversity_pvals",
        perm_mean_key: str = "spatiold_local_diversity_perm_mean",
    ) -> dict[str, pd.DataFrame | np.ndarray]:
        """Compute p-values, null mean, and permutation distribution in one run."""
        stats = compute_nd_permutation_stats(
            self.coords,
            self.labels,
            n_perm=n_perm,
            radii=radii,
            random_state=random_state,
            n_jobs=n_jobs,
            include_self=include_self,
            base=base,
            alternative=alternative,
        )
        if store:
            store_matrix_in_anndata(self.adata, stats["pvals"], key=pvals_key)
            store_matrix_in_anndata(self.adata, stats["perm_mean"], key=perm_mean_key)
        return stats

    def compute_permutation_mean(
        self,
        n_perm: int,
        *,
        radii: list[float] | tuple[float, ...] = DEFAULT_RADII,
        random_state: int = 42,
        n_jobs: int | None = None,
        include_self: bool = True,
        base: float = 2.0,
        store: bool = True,
        key: str = "spatiold_local_diversity_perm_mean",
    ) -> pd.DataFrame:
        """Compute permutation null mean matrix for local diversity."""
        perm_mean_df = compute_nd_permutation_mean(
            self.coords,
            self.labels,
            n_perm=n_perm,
            radii=radii,
            random_state=random_state,
            n_jobs=n_jobs,
            include_self=include_self,
            base=base,
        )
        if store:
            store_matrix_in_anndata(self.adata, perm_mean_df, key=key)
        return perm_mean_df

    def compute_permutation_distribution(
        self,
        n_perm: int,
        *,
        radii: list[float] | tuple[float, ...] = DEFAULT_RADII,
        random_state: int = 42,
        n_jobs: int | None = None,
        include_self: bool = True,
        base: float = 2.0,
    ) -> np.ndarray:
        """Compute full permutation distribution (n_perm, n_radii, n_cells)."""
        return compute_nd_permutation_distribution(
            self.coords,
            self.labels,
            n_perm=n_perm,
            radii=radii,
            random_state=random_state,
            n_jobs=n_jobs,
            include_self=include_self,
            base=base,
        )

    def summarize_local_diversity_by_cell_type(
        self,
        *,
        local_diversity_df: pd.DataFrame | None = None,
        local_diversity_key: str = "spatiold_local_diversity",
        cell_type_col: str | None = None,
        normalize_by: float | None = None,
        normalize_by_global_entropy: bool = True,
    ) -> pd.DataFrame:
        """Summarize local diversity by cell type/radius from object-held data."""
        if local_diversity_df is None:
            local_diversity_df = self.get_result(local_diversity_key)

        ct_col = self.label_key if cell_type_col is None else cell_type_col
        norm_val = normalize_by
        if normalize_by_global_entropy and normalize_by is None:
            norm_val = _compute_global_shannon_entropy(self.labels)

        return _summarize_local_diversity_by_cell_type(
            local_diversity_df,
            self.metadata,
            cell_type_col=ct_col,
            normalize_by=norm_val,
        )

    def compute_sample_vs_null_summary(
        self,
        permutation_distribution: np.ndarray,
        *,
        local_diversity_df: pd.DataFrame | None = None,
        local_diversity_key: str = "spatiold_local_diversity",
        normalize_by: float | None = None,
        normalize_by_global_entropy: bool = True,
        ci: tuple[float, float] = (2.5, 97.5),
    ) -> pd.DataFrame:
        """Compute sample-vs-null summary using object-held local diversity."""
        if local_diversity_df is None:
            local_diversity_df = self.get_result(local_diversity_key)

        norm_val = normalize_by
        if normalize_by_global_entropy and normalize_by is None:
            norm_val = _compute_global_shannon_entropy(self.labels)

        return _compute_sample_vs_null_summary(
            local_diversity_df,
            permutation_distribution,
            normalize_by=norm_val,
            ci=ci,
        )

    def cluster_local_diversity_profiles(
        self,
        *,
        local_diversity_df: pd.DataFrame | None = None,
        local_diversity_key: str = "spatiold_local_diversity",
        k_values: tuple[int, ...] | list[int] = (2, 3, 4, 5),
        scale: bool = True,
        random_state: int = 42,
        n_init: int = 20,
    ):
        """Cluster local-diversity profiles using object-held local diversity."""
        if local_diversity_df is None:
            local_diversity_df = self.get_result(local_diversity_key)

        return _cluster_local_diversity_profiles(
            local_diversity_df,
            k_values=k_values,
            scale=scale,
            random_state=random_state,
            n_init=n_init,
        )

    def build_significance_mask(
        self,
        *,
        pvals_df: pd.DataFrame | None = None,
        pvals_key: str = "spatiold_local_diversity_pvals",
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Build binary significance mask from p-values."""
        if pvals_df is None:
            pvals_df = self.get_result(pvals_key)
        return _build_significance_mask(pvals_df, alpha=alpha)

    def prepare_shared_components(
        self,
        *,
        response_matrix: pd.DataFrame | np.ndarray | None = None,
        local_diversity_key: str = "spatiold_local_diversity",
        radius_values: list[float] | tuple[float, ...] | None = None,
        cell_type_col: str | None = None,
        cell_id_col: str | None = None,
        reference_cell_type: str | None = None,
        radius_mode: str = "spline",
        n_radius_knots: int = 5,
        spline_degree: int = 3,
        poly_degree: int = 3,
        normalize_by: float | None = None,
        normalize_by_global_entropy: bool = True,
    ) -> dict[str, Any]:
        """Prepare shared components for gene-radius modeling using object data.

        By default, local-diversity response is normalized by object-level
        global Shannon entropy before regression.
        """
        if response_matrix is None:
            response_df = self.get_result(local_diversity_key)
            response_matrix = response_df.values
            if radius_values is None:
                radius_values = [float(c) for c in response_df.columns]
        if radius_values is None:
            raise ValueError("`radius_values` is required when `response_matrix` is not a DataFrame from the object.")

        ct_col = self.label_key if cell_type_col is None else cell_type_col

        return _prepare_shared_components(
            response_matrix=response_matrix,
            metadata_df=self.metadata,
            radius_values=radius_values,
            cell_type_col=ct_col,
            cell_id_col=cell_id_col,
            reference_cell_type=reference_cell_type,
            radius_mode=radius_mode,
            n_radius_knots=n_radius_knots,
            spline_degree=spline_degree,
            poly_degree=poly_degree,
            normalize_by=normalize_by,
            normalize_by_global_entropy=normalize_by_global_entropy,
        )

    def fit_slide_level_cell_type_radius_model(
        self,
        *,
        shared: dict[str, Any] | None = None,
        response_matrix: pd.DataFrame | np.ndarray | None = None,
        local_diversity_key: str = "spatiold_local_diversity",
        radius_values: list[float] | tuple[float, ...] | None = None,
        cell_type_col: str | None = None,
        cell_id_col: str | None = None,
        reference_cell_type: str | None = None,
        radius_mode: str = "spline",
        n_radius_knots: int = 5,
        spline_degree: int = 3,
        poly_degree: int = 3,
        normalize_by: float | None = None,
        normalize_by_global_entropy: bool = True,
        add_intercept: bool = True,
        cluster_robust: bool = True,
    ) -> dict[str, Any]:
        """Fit one slide-level model using cell-type and radius effects only."""
        shared_use = shared
        if shared_use is None:
            shared_use = self.prepare_shared_components(
                response_matrix=response_matrix,
                local_diversity_key=local_diversity_key,
                radius_values=radius_values,
                cell_type_col=cell_type_col,
                cell_id_col=cell_id_col,
                reference_cell_type=reference_cell_type,
                radius_mode=radius_mode,
                n_radius_knots=n_radius_knots,
                spline_degree=spline_degree,
                poly_degree=poly_degree,
                normalize_by=normalize_by,
                normalize_by_global_entropy=normalize_by_global_entropy,
            )

        return _fit_slide_level_cell_type_radius_model(
            shared_use,
            add_intercept=add_intercept,
            cluster_robust=cluster_robust,
        )

    def summarize_slide_level_cell_type_effects(
        self,
        fit_result: dict[str, Any],
        shared: dict[str, Any],
        *,
        include_reference: bool = True,
    ) -> pd.DataFrame:
        """Summarize per-cell-type effects from slide-level model fit."""
        return _summarize_slide_level_cell_type_effects(
            fit_result,
            shared,
            include_reference=include_reference,
        )

    def summarize_model_terms(self, fit_result: dict[str, Any]) -> pd.DataFrame:
        """Summarize model terms as coefficient/SE/p-value table."""
        return _summarize_model_terms(fit_result)

    def compute_svg_morans_i(
        self,
        expr_df: pd.DataFrame,
        *,
        x_col: str = "x",
        y_col: str = "y",
        k: int = 15,
        chunk_size: int = 64,
    ) -> pd.DataFrame:
        """Compute SVG ranking (Moran's I) using object coordinates."""
        coords_df = self.get_coords_df(x_col=x_col, y_col=y_col)
        return _compute_svg_morans_i(
            expr_df,
            coords_df,
            x_col=x_col,
            y_col=y_col,
            k=k,
            chunk_size=chunk_size,
        )
