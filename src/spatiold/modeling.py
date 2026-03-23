"""Regression helpers extracted and cleaned from the prototype workflow notebook."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LassoCV, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _effective_splits(n_splits: int, n_cells: int) -> int:
    return int(max(2, min(n_splits, n_cells)))


def _normalize_radius_df_columns(y_df: pd.DataFrame) -> pd.DataFrame:
    out = y_df.copy()
    out.columns = [float(c) for c in out.columns]
    return out


def _radius_code(
    radii: Sequence[float], center_radius: bool, radius_step_um: float
) -> tuple[np.ndarray, float, np.ndarray]:
    if radius_step_um <= 0:
        raise ValueError("`radius_step_um` must be > 0.")

    radius_vals = np.asarray([float(r) for r in radii], dtype=np.float32)
    r0 = 100.0 if center_radius else 0.0
    radius_vals_cs = (radius_vals - r0) / float(radius_step_um)
    return radius_vals, float(r0), radius_vals_cs


def top_abs_terms(coef_s: pd.Series, k: int = 10) -> pd.DataFrame:
    """Return top absolute coefficients while preserving sign in `coef`."""
    if k < 1:
        raise ValueError("`k` must be >= 1.")
    df = pd.DataFrame({"coef": coef_s, "abs_coef": coef_s.abs()})
    return df.sort_values("abs_coef", ascending=False).head(k)


def prepare_radius_design(
    X_gene_df: pd.DataFrame,
    cell_type_s: pd.Series,
    *,
    standardize_genes: bool = True,
) -> dict[str, Any]:
    """Prepare shared design matrices for radius-specific models.

    Returns a dictionary containing ``G``, ``C``, ``GCT`` and associated feature names.
    """
    if not X_gene_df.index.equals(cell_type_s.index):
        cell_type_s = cell_type_s.reindex(X_gene_df.index)

    if cell_type_s.isna().any():
        raise ValueError("`cell_type_s` contains missing values after index alignment.")

    gene_names = X_gene_df.columns.astype(str).tolist()
    X_gene = X_gene_df.values.astype(np.float32)

    gene_scaler: StandardScaler | None = None
    if standardize_genes:
        gene_scaler = StandardScaler(with_mean=True, with_std=True)
        X_gene = gene_scaler.fit_transform(X_gene).astype(np.float32)

    G = sparse.csr_matrix(X_gene)

    ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=True)
    C = ohe.fit_transform(cell_type_s.astype(str).to_frame(name="cell_type"))
    ct_names = ohe.get_feature_names_out(["cell_type"]).tolist()

    gct_blocks = []
    gct_names: list[str] = []
    for k, ct_name in enumerate(ct_names):
        gct_blocks.append(G.multiply(C[:, k]))
        gct_names.extend([f"{g}:{ct_name}" for g in gene_names])

    GCT = (
        sparse.hstack(gct_blocks, format="csr")
        if gct_blocks
        else sparse.csr_matrix((G.shape[0], 0), dtype=np.float32)
    )

    return {
        "G": G,
        "C": C,
        "GCT": GCT,
        "gene_scaler": gene_scaler,
        "gene_names": gene_names,
        "ct_names": ct_names,
        "gct_names": gct_names,
    }


def fit_lasso_for_celltype(
    ct_label: str,
    X_gene_df: pd.DataFrame,
    Y_ld_df: pd.DataFrame,
    ld_radii_list: Sequence[float],
    *,
    n_splits: int = 5,
    min_cells: int = 200,
    center_radius: bool = True,
    radius_step_um: float = 50.0,
    n_alphas: int = 30,
    eps: float = 1e-2,
    tol: float = 1e-3,
    max_iter: int = 3000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit cell-type-specific LASSO on long-form response across radii.

    Model:
        ``y_{i,r} ~ X_i + r + X_i:r``
    """
    n_cells = X_gene_df.shape[0]
    if n_cells < min_cells:
        return {
            "cell_type": ct_label,
            "skipped": True,
            "reason": f"Too few cells ({n_cells} < {min_cells})",
        }

    n_splits_eff = _effective_splits(n_splits, n_cells)
    if n_cells < 2:
        return {
            "cell_type": ct_label,
            "skipped": True,
            "reason": "Need at least 2 cells.",
        }

    Y_df = _normalize_radius_df_columns(Y_ld_df)

    gene_scaler = StandardScaler(with_mean=True, with_std=True)
    X_gene_scaled = gene_scaler.fit_transform(X_gene_df.values.astype(np.float32)).astype(np.float32)

    G = sparse.csr_matrix(X_gene_scaled)
    gene_names = X_gene_df.columns.astype(str).tolist()

    radius_vals, r0, radius_vals_cs = _radius_code(
        ld_radii_list, center_radius=center_radius, radius_step_um=radius_step_um
    )

    X_blocks = []
    y_blocks = []
    for rv, rv_cs in zip(radius_vals, radius_vals_cs):
        radius_col = sparse.csr_matrix(np.full((n_cells, 1), rv_cs, dtype=np.float32))
        X_r = sparse.hstack([G, radius_col, G * rv_cs], format="csr")
        X_blocks.append(X_r)
        y_blocks.append(Y_df.loc[:, float(rv)].to_numpy(dtype=np.float32))

    X_long = sparse.vstack(X_blocks, format="csr")
    y_long = np.concatenate(y_blocks)

    groups = np.tile(np.arange(n_cells), len(radius_vals_cs))
    cv_splits = list(GroupKFold(n_splits=n_splits_eff).split(np.zeros_like(y_long), y_long, groups))

    model = LassoCV(
        cv=cv_splits,
        n_alphas=n_alphas,
        eps=eps,
        tol=tol,
        max_iter=max_iter,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_long, y_long)

    feature_names = gene_names + ["radius_cs"] + [f"{g}:radius" for g in gene_names]
    coef_s = pd.Series(np.asarray(model.coef_).ravel(), index=feature_names)

    y_pred = model.predict(X_long)
    mse = float(np.mean((y_long - y_pred) ** 2))
    r2 = float(model.score(X_long, y_long))

    return {
        "cell_type": ct_label,
        "skipped": False,
        "model": model,
        "alpha_": float(model.alpha_),
        "coef_s": coef_s,
        "gene_scaler": gene_scaler,
        "gene_names": gene_names,
        "r0": r0,
        "radius_step_um": float(radius_step_um),
        "mse_in_sample": mse,
        "r2_in_sample": r2,
        "predict": y_pred,
        "true_label": y_long,
    }


def fit_ols_for_celltype(
    ct_label: str,
    X_gene_df: pd.DataFrame,
    Y_ld_df: pd.DataFrame,
    ld_radii_list: Sequence[float],
    *,
    n_splits: int = 5,
    min_cells: int = 200,
    center_radius: bool = True,
    radius_step_um: float = 50.0,
    standardize_genes: bool = True,
    fit_intercept: bool = True,
    return_cv_metrics: bool = True,
    use_ridge_if_singular: bool = True,
    ridge_alpha: float = 1e-6,
) -> dict[str, Any]:
    """Fit cell-type-specific OLS (or tiny Ridge fallback) on long-form response."""
    n_cells = X_gene_df.shape[0]
    if n_cells < min_cells:
        return {
            "cell_type": ct_label,
            "skipped": True,
            "reason": f"Too few cells ({n_cells} < {min_cells})",
        }

    if n_cells < 2:
        return {
            "cell_type": ct_label,
            "skipped": True,
            "reason": "Need at least 2 cells.",
        }

    n_splits_eff = _effective_splits(n_splits, n_cells)
    Y_df = _normalize_radius_df_columns(Y_ld_df)

    gene_names = X_gene_df.columns.astype(str).tolist()
    X_gene = X_gene_df.values.astype(np.float32)

    gene_scaler: StandardScaler | None = None
    if standardize_genes:
        gene_scaler = StandardScaler(with_mean=True, with_std=True)
        X_gene = gene_scaler.fit_transform(X_gene).astype(np.float32)

    radius_vals, r0, radius_vals_cs = _radius_code(
        ld_radii_list, center_radius=center_radius, radius_step_um=radius_step_um
    )

    X_blocks = []
    y_blocks = []
    for rv, rv_cs in zip(radius_vals, radius_vals_cs):
        radius_col = np.full((n_cells, 1), rv_cs, dtype=np.float32)
        X_r = np.hstack([X_gene, radius_col, X_gene * rv_cs]).astype(np.float32)
        X_blocks.append(X_r)
        y_blocks.append(Y_df.loc[:, float(rv)].to_numpy(dtype=np.float32))

    X_long = np.vstack(X_blocks)
    y_long = np.concatenate(y_blocks)

    groups = np.tile(np.arange(n_cells), len(radius_vals_cs))
    cv = GroupKFold(n_splits=n_splits_eff)

    cv_metrics = None
    if return_cv_metrics:
        mses = []
        r2s = []
        for tr, te in cv.split(X_long, y_long, groups=groups):
            try:
                m = LinearRegression(fit_intercept=fit_intercept)
                m.fit(X_long[tr], y_long[tr])
            except Exception:
                if not use_ridge_if_singular:
                    raise
                m = Ridge(alpha=ridge_alpha, fit_intercept=fit_intercept)
                m.fit(X_long[tr], y_long[tr])

            pred = m.predict(X_long[te])
            mses.append(mean_squared_error(y_long[te], pred))
            r2s.append(r2_score(y_long[te], pred))

        cv_metrics = {
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
        }

    try:
        model = LinearRegression(fit_intercept=fit_intercept).fit(X_long, y_long)
        model_name = "OLS"
    except Exception:
        if not use_ridge_if_singular:
            raise
        model = Ridge(alpha=ridge_alpha, fit_intercept=fit_intercept).fit(X_long, y_long)
        model_name = f"Ridge(alpha={ridge_alpha})"

    feature_names = gene_names + ["radius_cs"] + [f"{g}:radius" for g in gene_names]
    coef_s = pd.Series(np.asarray(model.coef_).ravel(), index=feature_names)

    pred_all = model.predict(X_long)
    mse_in = float(mean_squared_error(y_long, pred_all))
    r2_in = float(r2_score(y_long, pred_all))

    return {
        "cell_type": ct_label,
        "skipped": False,
        "model": model,
        "model_name": model_name,
        "coef_s": coef_s,
        "gene_scaler": gene_scaler,
        "gene_names": gene_names,
        "r0": r0,
        "radius_step_um": float(radius_step_um),
        "mse_in_sample": mse_in,
        "r2_in_sample": r2_in,
        "cv_metrics": cv_metrics,
        "predict": pred_all,
        "true_label": y_long,
    }


def fit_lasso_for_radius(
    radius_um: float,
    *,
    G: sparse.csr_matrix,
    C: sparse.csr_matrix,
    GCT: sparse.csr_matrix,
    gene_names: Sequence[str],
    ct_names: Sequence[str],
    gct_names: Sequence[str],
    y: np.ndarray,
    n_splits: int = 5,
    min_cells: int = 200,
    n_alphas: int = 30,
    eps: float = 1e-2,
    tol: float = 1e-3,
    max_iter: int = 3000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit radius-specific LASSO model: `y_i(r) ~ genes + cell_type + genes:cell_type`."""
    n_cells = y.shape[0]
    if n_cells < min_cells:
        return {
            "radius": float(radius_um),
            "skipped": True,
            "reason": f"Too few cells ({n_cells} < {min_cells})",
        }

    if n_cells < 2:
        return {
            "radius": float(radius_um),
            "skipped": True,
            "reason": "Need at least 2 cells.",
        }

    n_splits_eff = _effective_splits(n_splits, n_cells)

    X = sparse.hstack([G, C, GCT], format="csr")
    groups = np.arange(n_cells)
    cv_splits = list(GroupKFold(n_splits=n_splits_eff).split(np.zeros(n_cells), y, groups))

    model = LassoCV(
        cv=cv_splits,
        n_alphas=n_alphas,
        eps=eps,
        tol=tol,
        max_iter=max_iter,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X, y.astype(np.float32))

    feature_names = list(gene_names) + list(ct_names) + list(gct_names)
    coef_s = pd.Series(np.asarray(model.coef_).ravel(), index=feature_names)

    yhat = model.predict(X)
    return {
        "radius": float(radius_um),
        "skipped": False,
        "model": model,
        "coef_s": coef_s,
        "alpha_": float(model.alpha_),
        "mse_in_sample": float(np.mean((y - yhat) ** 2)),
        "r2_in_sample": float(model.score(X, y)),
    }


def fit_ols_for_radius(
    radius_um: float,
    *,
    G: sparse.csr_matrix,
    C: sparse.csr_matrix,
    GCT: sparse.csr_matrix,
    gene_names: Sequence[str],
    ct_names: Sequence[str],
    gct_names: Sequence[str],
    y: np.ndarray,
    n_splits: int = 5,
    min_cells: int = 200,
    fit_intercept: bool = True,
    return_cv_metrics: bool = True,
    use_ridge_if_singular: bool = True,
    ridge_alpha: float = 1e-6,
) -> dict[str, Any]:
    """Fit radius-specific OLS (or tiny Ridge fallback) model."""
    n_cells = y.shape[0]
    if n_cells < min_cells:
        return {
            "radius": float(radius_um),
            "skipped": True,
            "reason": f"Too few cells ({n_cells} < {min_cells})",
        }

    if n_cells < 2:
        return {
            "radius": float(radius_um),
            "skipped": True,
            "reason": "Need at least 2 cells.",
        }

    n_splits_eff = _effective_splits(n_splits, n_cells)

    X = sparse.hstack([G, C, GCT], format="csr").toarray().astype(np.float32)
    y = y.astype(np.float32)

    groups = np.arange(n_cells)
    cv = GroupKFold(n_splits=n_splits_eff)

    cv_metrics = None
    if return_cv_metrics:
        mses = []
        r2s = []
        for tr, te in cv.split(X, y, groups=groups):
            try:
                m = LinearRegression(fit_intercept=fit_intercept)
                m.fit(X[tr], y[tr])
            except Exception:
                if not use_ridge_if_singular:
                    raise
                m = Ridge(alpha=ridge_alpha, fit_intercept=fit_intercept)
                m.fit(X[tr], y[tr])

            pred = m.predict(X[te])
            mses.append(mean_squared_error(y[te], pred))
            r2s.append(r2_score(y[te], pred))

        cv_metrics = {
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
        }

    try:
        model = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
        model_name = "OLS"
    except Exception:
        if not use_ridge_if_singular:
            raise
        model = Ridge(alpha=ridge_alpha, fit_intercept=fit_intercept).fit(X, y)
        model_name = f"Ridge(alpha={ridge_alpha})"

    feature_names = list(gene_names) + list(ct_names) + list(gct_names)
    coef_s = pd.Series(np.asarray(model.coef_).ravel(), index=feature_names)

    pred_all = model.predict(X)
    return {
        "radius": float(radius_um),
        "skipped": False,
        "model": model,
        "model_name": model_name,
        "coef_s": coef_s,
        "mse_in_sample": float(mean_squared_error(y, pred_all)),
        "r2_in_sample": float(r2_score(y, pred_all)),
        "cv_metrics": cv_metrics,
    }
