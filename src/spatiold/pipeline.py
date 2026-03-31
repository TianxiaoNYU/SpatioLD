"""Higher-level analysis pipeline utilities based on the updated SlideTag workflow."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


def compute_global_shannon_entropy(
    labels: pd.Series | np.ndarray | Sequence[object],
    *,
    base: float = 2.0,
) -> float:
    """Compute global Shannon entropy of label composition."""
    labels_arr = np.asarray(labels)
    _, counts = np.unique(labels_arr.astype(str), return_counts=True)
    probs = counts / counts.sum()
    log_probs = np.log(probs) / np.log(base)
    return float(-(probs * log_probs).sum())


def preprocess_expression_matrix(
    expr_df: pd.DataFrame,
    *,
    min_fraction_expressed: float = 0.02,
    min_genes_per_cell: int = 50,
) -> pd.DataFrame:
    """Filter expression matrix similarly to the SlideTag workflow.

    Input and output shape are ``cells x genes``.
    """
    if not 0 <= min_fraction_expressed <= 1:
        raise ValueError("`min_fraction_expressed` must be in [0, 1].")
    if min_genes_per_cell < 0:
        raise ValueError("`min_genes_per_cell` must be >= 0.")

    out = expr_df.copy()
    gene_mask = (out > 0).mean(axis=0) >= min_fraction_expressed
    out = out.loc[:, gene_mask]

    if min_genes_per_cell > 0:
        cell_mask = (out > 0).sum(axis=1) >= min_genes_per_cell
        out = out.loc[cell_mask, :]

    return out


def align_expression_and_metadata(
    expr_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align expression and metadata on shared cell IDs using index intersection."""
    common = expr_df.index.astype(str).intersection(metadata_df.index.astype(str))
    expr_aligned = expr_df.copy()
    meta_aligned = metadata_df.copy()
    expr_aligned.index = expr_aligned.index.astype(str)
    meta_aligned.index = meta_aligned.index.astype(str)
    return expr_aligned.loc[common], meta_aligned.loc[common]


def summarize_local_diversity_by_cell_type(
    local_diversity_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    *,
    cell_type_col: str = "cell_type",
    normalize_by: float | None = None,
) -> pd.DataFrame:
    """Summarize mean/std local diversity by cell type and radius."""
    if cell_type_col not in metadata_df.columns:
        raise KeyError(f"`{cell_type_col}` not found in metadata_df.")

    ld = local_diversity_df.copy()
    ld.index = ld.index.astype(str)
    ld.columns = [float(c) for c in ld.columns]

    meta = metadata_df.copy()
    meta.index = meta.index.astype(str)

    common = ld.index.intersection(meta.index)
    ld = ld.loc[common]
    ct = meta.loc[common, cell_type_col].astype(str)

    long_df = ld.stack().rename("local_diversity").reset_index()
    long_df = long_df.rename(
        columns={
            long_df.columns[0]: "cell_id",
            long_df.columns[1]: "radius",
        }
    )
    long_df["cell_type"] = long_df["cell_id"].map(ct)

    summary = (
        long_df.groupby(["cell_type", "radius"], as_index=False)
        .agg(mean=("local_diversity", "mean"), std=("local_diversity", "std"), n=("local_diversity", "size"))
        .sort_values(["cell_type", "radius"])
        .reset_index(drop=True)
    )

    summary["std"] = summary["std"].fillna(0.0)
    if normalize_by is not None:
        summary["mean"] = summary["mean"] / normalize_by
        summary["std"] = summary["std"] / normalize_by

    return summary


def compute_sample_vs_null_summary(
    local_diversity_df: pd.DataFrame,
    permutation_distribution: np.ndarray,
    *,
    normalize_by: float | None = None,
    ci: tuple[float, float] = (2.5, 97.5),
) -> pd.DataFrame:
    """Summarize sample vs permutation-null diversity across radii.

    Parameters
    ----------
    local_diversity_df
        DataFrame with shape ``(n_cells, n_radii)``.
    permutation_distribution
        Array with shape ``(n_perm, n_radii, n_cells)``.
    """
    ld = local_diversity_df.copy()
    ld.columns = [float(c) for c in ld.columns]
    obs = ld.to_numpy(dtype=float).T  # (n_radii, n_cells)

    perm = np.asarray(permutation_distribution, dtype=float)
    if perm.ndim != 3:
        raise ValueError("`permutation_distribution` must be 3D (n_perm, n_radii, n_cells).")
    if perm.shape[1:] != obs.shape:
        raise ValueError(
            "`permutation_distribution` shape mismatch. Expected "
            f"(n_perm, {obs.shape[0]}, {obs.shape[1]}), got {perm.shape}."
        )

    sample_mean = obs.mean(axis=1)
    sample_std = obs.std(axis=1)

    null_mean = perm.mean(axis=(0, 2))
    null_perm_means = perm.mean(axis=2)  # (n_perm, n_radii)
    q_low, q_high = np.percentile(null_perm_means, ci, axis=0)

    out = pd.DataFrame(
        {
            "radius": ld.columns,
            "sample_mean": sample_mean,
            "sample_std": sample_std,
            "null_mean": null_mean,
            "null_ci_low": q_low,
            "null_ci_high": q_high,
        }
    )

    if normalize_by is not None:
        for col in ["sample_mean", "sample_std", "null_mean", "null_ci_low", "null_ci_high"]:
            out[col] = out[col] / normalize_by

    return out


def cluster_local_diversity_profiles(
    local_diversity_df: pd.DataFrame,
    *,
    k_values: Sequence[int] = (2, 3, 4, 5),
    scale: bool = True,
    random_state: int = 42,
    n_init: int = 20,
) -> tuple[pd.DataFrame, dict[int, KMeans]]:
    """Cluster local-diversity profiles of cells across radii using KMeans."""
    X = local_diversity_df.to_numpy(dtype=float)
    if scale:
        X = StandardScaler().fit_transform(X)

    labels_df = pd.DataFrame(index=local_diversity_df.index.astype(str))
    models: dict[int, KMeans] = {}

    for k in k_values:
        model = KMeans(n_clusters=int(k), random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X)
        models[int(k)] = model
        labels_df[f"ld_kmeans_k{k}"] = labels.astype(str)

    return labels_df, models


def build_significance_mask(
    pvals_df: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Return binary significance mask (`1` if p < alpha else `0`)."""
    return (pvals_df < alpha).astype(int)


def make_spline_basis(
    radius_values: Sequence[float],
    *,
    n_knots: int = 5,
    degree: int = 3,
    include_bias: bool = False,
) -> tuple[np.ndarray, SplineTransformer]:
    """Create a spline basis for a scalar radius grid."""
    r = np.asarray(radius_values, dtype=float).reshape(-1, 1)
    transformer = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        include_bias=include_bias,
    )
    basis = transformer.fit_transform(r)
    return basis, transformer


def prepare_shared_components(
    response_matrix: np.ndarray | pd.DataFrame,
    metadata_df: pd.DataFrame,
    radius_values: Sequence[float],
    *,
    cell_type_col: str = "cell_type",
    cell_id_col: str | None = None,
    reference_cell_type: str | None = None,
    radius_mode: str = "spline",
    n_radius_knots: int = 5,
    spline_degree: int = 3,
    poly_degree: int = 3,
    covariate_cols: Sequence[str] | None = None,
    normalize_by: float | None = None,
    normalize_by_global_entropy: bool = True,
) -> dict[str, Any]:
    """Prepare shared design components for per-gene radius model fitting.

    By default, response values are normalized by sample-level global entropy
    of `cell_type_col` to improve comparability across samples.
    """
    Y = np.asarray(response_matrix, dtype=float)
    n_cells, n_radii = Y.shape
    radius_values = np.asarray(radius_values, dtype=float)

    if len(radius_values) != n_radii:
        raise ValueError("`radius_values` must match number of response columns.")
    if len(metadata_df) != n_cells:
        raise ValueError("`metadata_df` must have same row count as `response_matrix`.")
    if cell_type_col not in metadata_df.columns:
        raise KeyError(f"`{cell_type_col}` not found in metadata_df.")

    meta = metadata_df.copy()
    cell_types = meta[cell_type_col].astype(str)

    covariate_cols_use: list[str] = []
    covariate_feature_names: list[str] = []
    covariates_long = np.empty((n_cells * n_radii, 0), dtype=float)
    if covariate_cols is not None:
        covariate_cols_use = [str(c) for c in covariate_cols]
        missing_covars = [c for c in covariate_cols_use if c not in meta.columns]
        if missing_covars:
            raise KeyError(f"Covariate columns not found in metadata_df: {missing_covars}")

        cov_df = meta.loc[:, covariate_cols_use].apply(pd.to_numeric, errors="coerce")
        bad_covars = [c for c in cov_df.columns if cov_df[c].isna().any()]
        if bad_covars:
            raise ValueError(
                "Covariate columns must be numeric and non-missing. "
                f"Problematic columns: {bad_covars}"
            )

        covariates = cov_df.to_numpy(dtype=float)
        covariates_long = np.repeat(covariates, repeats=n_radii, axis=0)
        covariate_feature_names = [f"covariate_{c}" for c in covariate_cols_use]

    norm_factor = normalize_by
    if norm_factor is not None:
        norm_factor = float(norm_factor)
        if not np.isfinite(norm_factor) or norm_factor <= 0:
            raise ValueError("`normalize_by` must be a finite positive value.")
    elif normalize_by_global_entropy:
        entropy = compute_global_shannon_entropy(cell_types)
        if np.isfinite(entropy) and entropy > 0:
            norm_factor = float(entropy)

    if norm_factor is not None:
        Y = Y / norm_factor

    if cell_id_col is None or cell_id_col not in meta.columns:
        meta["_cell_id_internal"] = np.arange(n_cells)
        cell_id_col_use = "_cell_id_internal"
    else:
        cell_id_col_use = cell_id_col

    if reference_cell_type is None:
        reference_cell_type = cell_types.value_counts().idxmax()

    ordered_ct = [reference_cell_type] + [c for c in pd.unique(cell_types) if c != reference_cell_type]

    ct_encoder = OneHotEncoder(
        drop="first",
        sparse_output=False,
        dtype=float,
        categories=[ordered_ct],
        handle_unknown="ignore",
    )
    ct_encoded = ct_encoder.fit_transform(
        pd.DataFrame({cell_type_col: pd.Categorical(cell_types, categories=ordered_ct)})
    )
    ct_feature_names = list(ct_encoder.get_feature_names_out([cell_type_col]))

    if radius_mode == "spline":
        radius_basis, radius_transformer = make_spline_basis(
            radius_values,
            n_knots=n_radius_knots,
            degree=spline_degree,
            include_bias=False,
        )
        radius_feature_names = [f"spline_radius_{k}" for k in range(radius_basis.shape[1])]
    elif radius_mode == "poly":
        r = radius_values.astype(float)
        r_center = r.mean()
        r_scale = r.std(ddof=0)
        if r_scale == 0:
            r_scale = 1.0
        rz = (r - r_center) / r_scale
        radius_basis = np.column_stack([rz**d for d in range(1, poly_degree + 1)])
        radius_transformer = {
            "mode": "poly",
            "center": r_center,
            "scale": r_scale,
            "degree": poly_degree,
        }
        radius_feature_names = [f"poly_radius_deg{d}" for d in range(1, poly_degree + 1)]
    else:
        raise ValueError("`radius_mode` must be 'spline' or 'poly'.")

    y_long = Y.reshape(-1)
    ct_long = np.repeat(ct_encoded, repeats=n_radii, axis=0)
    radius_long = np.tile(radius_basis, (n_cells, 1))
    groups_long = np.repeat(meta[cell_id_col_use].values, repeats=n_radii)

    return {
        "Y": Y,
        "y_long": y_long,
        "n_cells": n_cells,
        "n_radii": n_radii,
        "radius_values": radius_values,
        "cell_type_col": cell_type_col,
        "cell_type_levels": ordered_ct,
        "groups_long": groups_long,
        "covariate_cols": covariate_cols_use,
        "covariates_long": covariates_long,
        "covariate_feature_names": covariate_feature_names,
        "ct_long": ct_long,
        "ct_feature_names": ct_feature_names,
        "reference_cell_type": reference_cell_type,
        "radius_basis": radius_basis,
        "radius_long": radius_long,
        "radius_feature_names": radius_feature_names,
        "radius_mode": radius_mode,
        "radius_transformer": radius_transformer,
        "response_normalization_factor": norm_factor,
    }


def fit_single_gene_radius_model(
    gene_values: np.ndarray | pd.Series,
    shared: dict[str, Any],
    *,
    add_intercept: bool = True,
    cluster_robust: bool = True,
) -> dict[str, Any]:
    """Fit one-gene model from the SlideTag pipeline.

    Model:
        ``E_{i,r} = beta * x_i + cell_type + f(r) + error``
    """
    import statsmodels.api as sms

    x = np.asarray(gene_values, dtype=float).reshape(-1)
    if len(x) != shared["n_cells"]:
        raise ValueError("`gene_values` must have length n_cells.")

    n_radii = shared["n_radii"]
    x_long = np.repeat(x, repeats=n_radii).reshape(-1, 1)

    covariates_long = np.asarray(
        shared.get("covariates_long", np.empty((shared["y_long"].shape[0], 0), dtype=float)),
        dtype=float,
    )
    if covariates_long.ndim != 2 or covariates_long.shape[0] != shared["y_long"].shape[0]:
        raise ValueError("`shared['covariates_long']` must be a 2D array with one row per long-form sample.")
    covariate_feature_names = list(shared.get("covariate_feature_names", []))

    X = np.hstack([x_long, covariates_long, shared["ct_long"], shared["radius_long"]])
    feature_names = (
        ["gene"]
        + covariate_feature_names
        + shared["ct_feature_names"]
        + shared["radius_feature_names"]
    )

    if add_intercept:
        X = sms.add_constant(X, has_constant="add")
        feature_names = ["const"] + feature_names

    model = sms.OLS(shared["y_long"], X)
    if cluster_robust:
        fit_res = model.fit(cov_type="cluster", cov_kwds={"groups": shared["groups_long"]})
    else:
        fit_res = model.fit()

    coef = pd.Series(fit_res.params, index=feature_names)
    se = pd.Series(fit_res.bse, index=feature_names)
    pval = pd.Series(fit_res.pvalues, index=feature_names)

    return {
        "fit": fit_res,
        "coef": coef,
        "se": se,
        "pval": pval,
        "feature_names": feature_names,
    }


def fit_slide_level_cell_type_radius_model(
    shared: dict[str, Any],
    *,
    add_intercept: bool = True,
    cluster_robust: bool = True,
) -> dict[str, Any]:
    """Fit one slide-level model using only cell type and radius terms.

    Model:
        ``E_{i,r} = cell_type_i + f(r) + error``
    """
    import statsmodels.api as sms

    covariates_long = np.asarray(
        shared.get("covariates_long", np.empty((shared["y_long"].shape[0], 0), dtype=float)),
        dtype=float,
    )
    if covariates_long.ndim != 2 or covariates_long.shape[0] != shared["y_long"].shape[0]:
        raise ValueError("`shared['covariates_long']` must be a 2D array with one row per long-form sample.")
    covariate_feature_names = list(shared.get("covariate_feature_names", []))

    X = np.hstack([covariates_long, shared["ct_long"], shared["radius_long"]])
    feature_names = covariate_feature_names + shared["ct_feature_names"] + shared["radius_feature_names"]

    if add_intercept:
        X = sms.add_constant(X, has_constant="add")
        feature_names = ["const"] + feature_names

    model = sms.OLS(shared["y_long"], X)
    if cluster_robust:
        fit_res = model.fit(cov_type="cluster", cov_kwds={"groups": shared["groups_long"]})
    else:
        fit_res = model.fit()

    coef = pd.Series(fit_res.params, index=feature_names)
    se = pd.Series(fit_res.bse, index=feature_names)
    pval = pd.Series(fit_res.pvalues, index=feature_names)

    return {
        "fit": fit_res,
        "coef": coef,
        "se": se,
        "pval": pval,
        "feature_names": feature_names,
    }


def summarize_model_terms(fit_result: dict[str, Any]) -> pd.DataFrame:
    """Summarize model coefficients with SE/p-value statistics."""
    coef = fit_result["coef"]
    se = fit_result["se"]
    pval = fit_result["pval"]

    df = pd.DataFrame(
        {
            "term": coef.index.astype(str),
            "beta": coef.values.astype(float),
            "se": se.values.astype(float),
            "pval": pval.values.astype(float),
        }
    )
    df["t"] = np.where(df["se"] != 0, df["beta"] / df["se"], np.nan)
    return df


def summarize_slide_level_cell_type_effects(
    fit_result: dict[str, Any],
    shared: dict[str, Any],
    *,
    include_reference: bool = True,
) -> pd.DataFrame:
    """Extract slide-level cell-type effects from fitted cell-type/radius model."""
    coef = fit_result["coef"]
    se = fit_result["se"]
    pval = fit_result["pval"]

    ref = str(shared["reference_cell_type"])
    levels = [str(c) for c in shared["cell_type_levels"]]
    feature_names = [str(c) for c in shared["ct_feature_names"]]

    rows: list[dict[str, Any]] = []
    if include_reference:
        rows.append(
            {
                "cell_type": ref,
                "feature": "reference",
                "is_reference": True,
                "beta_cell_type": 0.0,
                "se_cell_type": np.nan,
                "pval_cell_type": np.nan,
                "t_cell_type": np.nan,
            }
        )

    for ct, feature in zip(levels[1:], feature_names, strict=False):
        beta = float(coef[feature])
        beta_se = float(se[feature])
        rows.append(
            {
                "cell_type": ct,
                "feature": feature,
                "is_reference": False,
                "beta_cell_type": beta,
                "se_cell_type": beta_se,
                "pval_cell_type": float(pval[feature]),
                "t_cell_type": beta / beta_se if beta_se != 0 else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "cell_type",
                "feature",
                "is_reference",
                "beta_cell_type",
                "se_cell_type",
                "pval_cell_type",
                "t_cell_type",
            ]
        )
    return out.reset_index(drop=True)


def reconstruct_radius_effect(
    fit_result: dict[str, Any],
    shared: dict[str, Any],
    *,
    include_intercept: bool = False,
) -> np.ndarray:
    """Reconstruct fitted radius function `f(r)` from fitted coefficients."""
    coef = fit_result["coef"]
    radius_coef = np.array([coef[name] for name in shared["radius_feature_names"]], dtype=float)
    f_r = shared["radius_basis"] @ radius_coef

    if include_intercept and "const" in coef.index:
        f_r = f_r + float(coef["const"])

    return f_r


def fit_all_genes(
    expr_df: pd.DataFrame,
    shared: dict[str, Any],
    *,
    cluster_robust: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Fit all genes one by one and return summary table + full fit objects."""
    if len(expr_df) != shared["n_cells"]:
        raise ValueError("`expr_df` must align with response matrix cell order.")

    summaries = []
    fit_objects: dict[str, dict[str, Any]] = {}

    n_genes = len(expr_df.columns)
    for k, gene in enumerate(expr_df.columns, start=1):
        verbose_n = max(100, n_genes // 20)

        if verbose and (k == 1 or k % verbose_n == 0 or k == n_genes):
            print(f"Fitting gene {k}/{n_genes}: {gene}")

        fit_res = fit_single_gene_radius_model(
            gene_values=expr_df[gene].values,  # type: ignore
            shared=shared,
            cluster_robust=cluster_robust,
        )
        fit_objects[str(gene)] = fit_res

        coef = fit_res["coef"]
        se = fit_res["se"]
        pval = fit_res["pval"]

        gene_coef = float(coef["gene"])
        gene_se = float(se["gene"])

        summaries.append(
            {
                "gene": str(gene),
                "beta_gene": gene_coef,
                "se_gene": gene_se,
                "pval_gene": float(pval["gene"]),
                "t_gene": gene_coef / gene_se if gene_se != 0 else np.nan,
                "r2": float(fit_res["fit"].rsquared),
                "adj_r2": float(fit_res["fit"].rsquared_adj),
                "aic": float(fit_res["fit"].aic),
                "bic": float(fit_res["fit"].bic),
            }
        )

    summary_df = pd.DataFrame(summaries).sort_values("pval_gene").reset_index(drop=True)
    return summary_df, fit_objects


def compute_svg_morans_i(
    expr_df: pd.DataFrame,
    coords_df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    k: int = 15,
    chunk_size: int = 64,
) -> pd.DataFrame:
    """Compute spatially variable genes via vectorized Moran's I on kNN graph."""
    expr = expr_df.copy()
    coords = coords_df.copy()

    expr.index = expr.index.astype(str)
    coords.index = coords.index.astype(str)
    common = expr.index.intersection(coords.index)
    expr = expr.loc[common]
    coords = coords.loc[common]

    xy = coords[[x_col, y_col]].to_numpy(dtype=np.float64)
    n = xy.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 cells to compute Moran's I.")
    k = int(min(k, n - 1))

    tree = KDTree(xy)
    _, nn_idx = tree.query(xy, k=k + 1)
    nn_idx = nn_idx[:, 1:]

    rows = np.repeat(np.arange(n), k)
    cols = nn_idx.reshape(-1)
    vals = np.ones(rows.shape[0], dtype=np.float32)

    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    W = sparse.diags(1.0 / np.maximum(row_sums, 1e-12)) @ W
    S0 = float(W.sum())

    X = expr.to_numpy(dtype=np.float32)
    Z = X - X.mean(axis=0, keepdims=True)
    den = (Z**2).sum(axis=0)

    I = np.zeros(X.shape[1], dtype=np.float64)
    for j0 in range(0, X.shape[1], chunk_size):
        j1 = min(j0 + chunk_size, X.shape[1])
        Zc = Z[:, j0:j1]
        WZc = W @ Zc
        num = (Zc * WZc).sum(axis=0)
        I[j0:j1] = (n / S0) * (num / np.maximum(den[j0:j1], 1e-12))

    return (
        pd.DataFrame({"gene": expr.columns.astype(str), "moran_I": I})
        .sort_values("moran_I", ascending=False)
        .reset_index(drop=True)
    )


def compute_hvg_scanpy(
    expr_df: pd.DataFrame,
    *,
    n_top_genes: int = 100,
    flavor: str = "seurat",
) -> pd.DataFrame:
    """Compute highly variable genes using Scanpy.

    Returns the subset of ``adata.var`` where `highly_variable` is True.
    """
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("scanpy is required for `compute_hvg_scanpy`.") from exc

    adata = sc.AnnData(expr_df.astype(np.float32).copy())
    adata.obs_names = expr_df.index.astype(str)
    adata.var_names = expr_df.columns.astype(str)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_top = min(int(n_top_genes), adata.n_vars)
    sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes=n_top)

    hvg_df = (
        adata.var.loc[adata.var["highly_variable"], ["means", "dispersions", "dispersions_norm"]]
        .sort_values("dispersions_norm", ascending=False)
        .copy()
    )
    hvg_df.index = hvg_df.index.astype(str)
    return hvg_df
