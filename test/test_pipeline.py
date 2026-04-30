from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatiold import (
    compute_global_shannon_entropy,
    compute_local_diversity_multi_radius,
    compute_nd_permutation_distribution,
    compute_nd_permutation_stats,
    compute_sample_vs_null_summary,
    compute_sample_vs_null_summary_from_permutation_means,
    compute_svg_morans_i,
    fit_all_genes,
    fit_joint_gene_radius_model,
    fit_slide_level_cell_type_radius_model,
    fit_single_gene_radius_model,
    prepare_shared_components,
    summarize_model_terms,
    summarize_slide_level_cell_type_effects,
    summarize_local_diversity_by_cell_type,
)


def _make_small_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 36
    coords = pd.DataFrame(
        {
            "x": rng.uniform(0, 100, size=n),
            "y": rng.uniform(0, 100, size=n),
        },
        index=[f"c{i}" for i in range(n)],
    )
    labels = pd.Series(
        rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.4, 0.2]),
        index=coords.index,
        name="cell_type",
    )
    meta = coords.copy()
    meta["cell_type"] = labels
    meta["cell_size"] = rng.uniform(50, 200, size=n)
    return coords, labels, meta


def test_pipeline_summaries_and_null_curve() -> None:
    coords, labels, meta = _make_small_dataset()
    ld = compute_local_diversity_multi_radius(coords, labels, radii=[10, 20, 30])

    entropy = compute_global_shannon_entropy(labels)
    summary_ct = summarize_local_diversity_by_cell_type(ld, meta, normalize_by=entropy)
    assert {"cell_type", "radius", "mean", "std", "n"}.issubset(summary_ct.columns)

    dist = compute_nd_permutation_distribution(coords, labels, n_perm=8, radii=[10, 20, 30], n_jobs=1)
    summary_null = compute_sample_vs_null_summary(ld, dist, normalize_by=entropy)
    assert {"radius", "sample_mean", "null_mean", "null_ci_low", "null_ci_high"}.issubset(summary_null.columns)


def test_sample_vs_null_summary_from_permutation_means_matches_distribution_summary() -> None:
    coords, labels, _ = _make_small_dataset()
    ld = compute_local_diversity_multi_radius(coords, labels, radii=[10, 20, 30])
    stats = compute_nd_permutation_stats(
        coords,
        labels,
        n_perm=8,
        radii=[10, 20, 30],
        n_jobs=1,
        return_distribution=True,
        return_permutation_means=True,
    )

    from_dist = compute_sample_vs_null_summary(ld, stats["distribution"])
    from_means = compute_sample_vs_null_summary_from_permutation_means(ld, stats["permutation_means"])

    pd.testing.assert_frame_equal(from_dist, from_means)


def test_gene_radius_model_and_svg() -> None:
    pytest.importorskip("statsmodels")

    coords, labels, meta = _make_small_dataset()
    ld = compute_local_diversity_multi_radius(coords, labels, radii=[10, 20, 30])

    rng = np.random.default_rng(1)
    expr = pd.DataFrame(
        rng.normal(size=(ld.shape[0], 8)),
        index=ld.index,
        columns=[f"g{i}" for i in range(8)],
    )

    shared = prepare_shared_components(
        response_matrix=ld.values,
        metadata_df=meta.loc[ld.index],
        radius_values=[10, 20, 30],
        cell_type_col="cell_type",
        radius_mode="poly",
        poly_degree=2,
        covariate_cols=["cell_size"],
    )

    fit = fit_single_gene_radius_model(expr["g0"].values, shared)
    assert "gene" in fit["coef"].index
    assert "covariate_cell_size" in fit["coef"].index

    svg_df = compute_svg_morans_i(expr, coords)
    assert list(svg_df.columns) == ["gene", "moran_I"]
    assert svg_df.shape[0] == expr.shape[1]


def test_fit_all_genes_can_skip_retaining_full_fit_objects() -> None:
    pytest.importorskip("statsmodels")

    coords, labels, meta = _make_small_dataset()
    ld = compute_local_diversity_multi_radius(coords, labels, radii=[10, 20, 30])

    rng = np.random.default_rng(11)
    expr = pd.DataFrame(
        rng.normal(size=(ld.shape[0], 4)),
        index=ld.index,
        columns=[f"g{i}" for i in range(4)],
    )

    shared = prepare_shared_components(
        response_matrix=ld.values,
        metadata_df=meta.loc[ld.index],
        radius_values=[10, 20, 30],
        cell_type_col="cell_type",
        radius_mode="poly",
        poly_degree=2,
    )

    results_df, fit_objects = fit_all_genes(
        expr,
        shared,
        verbose=False,
        store_fit_objects=False,
    )

    assert results_df.shape[0] == expr.shape[1]
    assert fit_objects == {}


def test_joint_gene_radius_model_recovers_known_coefficients() -> None:
    pytest.importorskip("statsmodels")

    rng = np.random.default_rng(23)
    n_cells = 45
    radii = [10.0, 25.0, 40.0, 55.0]
    cell_ids = [f"c{i}" for i in range(n_cells)]

    expr = pd.DataFrame(
        rng.normal(size=(n_cells, 3)),
        index=cell_ids,
        columns=["g0", "g1", "g2"],
    )
    meta = pd.DataFrame(
        {
            "cell_type": ["A"] * 20 + ["B"] * 15 + ["C"] * 10,
            "cell_size": rng.uniform(50, 200, size=n_cells),
        },
        index=cell_ids,
    )

    shared_template = prepare_shared_components(
        response_matrix=np.zeros((n_cells, len(radii)), dtype=float),
        metadata_df=meta,
        radius_values=radii,
        cell_type_col="cell_type",
        reference_cell_type="A",
        radius_mode="poly",
        poly_degree=2,
        covariate_cols=["cell_size"],
        normalize_by_global_entropy=False,
    )

    gene_beta = np.array([1.4, -0.9, 0.6], dtype=float)
    cov_beta = np.array([0.03], dtype=float)
    ct_beta = np.array([0.8, -0.5], dtype=float)
    radius_beta = np.array([0.35, -0.15], dtype=float)
    intercept = 0.7

    x_long = np.repeat(expr.to_numpy(dtype=float), repeats=len(radii), axis=0)
    y_long = (
        intercept
        + x_long @ gene_beta
        + shared_template["covariates_long"] @ cov_beta
        + shared_template["ct_long"] @ ct_beta
        + shared_template["radius_long"] @ radius_beta
        + rng.normal(scale=0.03, size=n_cells * len(radii))
    )
    Y = y_long.reshape(n_cells, len(radii))

    shared = prepare_shared_components(
        response_matrix=Y,
        metadata_df=meta,
        radius_values=radii,
        cell_type_col="cell_type",
        reference_cell_type="A",
        radius_mode="poly",
        poly_degree=2,
        covariate_cols=["cell_size"],
        normalize_by_global_entropy=False,
    )

    fit = fit_joint_gene_radius_model(
        expr,
        shared,
        cluster_robust=False,
    )

    gene_summary = fit["gene_summary"].set_index("gene")
    np.testing.assert_allclose(gene_summary.loc["g0", "beta_gene"], gene_beta[0], atol=0.05)
    np.testing.assert_allclose(gene_summary.loc["g1", "beta_gene"], gene_beta[1], atol=0.05)
    np.testing.assert_allclose(gene_summary.loc["g2", "beta_gene"], gene_beta[2], atol=0.05)
    assert "covariate_cell_size" in fit["coef"].index
    assert any(term.startswith("cell_type_") for term in fit["coef"].index)
    assert any(term.startswith("poly_radius_") for term in fit["coef"].index)


@pytest.mark.parametrize("cluster_robust", [False, True])
def test_fit_all_genes_batch_matches_single_loop(cluster_robust: bool) -> None:
    pytest.importorskip("statsmodels")

    coords, labels, meta = _make_small_dataset()
    ld = compute_local_diversity_multi_radius(coords, labels, radii=[10, 20, 30, 40])

    rng = np.random.default_rng(19)
    expr = pd.DataFrame(
        rng.normal(size=(ld.shape[0], 6)),
        index=ld.index,
        columns=[f"g{i}" for i in range(6)],
    )

    shared = prepare_shared_components(
        response_matrix=ld.values,
        metadata_df=meta.loc[ld.index],
        radius_values=[10, 20, 30, 40],
        cell_type_col="cell_type",
        radius_mode="poly",
        poly_degree=3,
        covariate_cols=["cell_size"],
    )

    single_df, _ = fit_all_genes(
        expr,
        shared,
        cluster_robust=cluster_robust,
        verbose=False,
        store_fit_objects=False,
        method="single",
    )
    batch_df, _ = fit_all_genes(
        expr,
        shared,
        cluster_robust=cluster_robust,
        verbose=False,
        store_fit_objects=False,
        method="batch",
        chunk_size=2,
    )

    single_sorted = single_df.sort_values("gene").reset_index(drop=True)
    batch_sorted = batch_df.sort_values("gene").reset_index(drop=True)

    assert single_sorted["gene"].tolist() == batch_sorted["gene"].tolist()
    for col in ["beta_gene", "se_gene", "pval_gene", "t_gene", "r2", "adj_r2", "aic", "bic"]:
        np.testing.assert_allclose(
            single_sorted[col].to_numpy(),
            batch_sorted[col].to_numpy(),
            rtol=1e-5,
            atol=1e-7,
        )


def test_prepare_shared_components_entropy_normalization_controls() -> None:
    meta = pd.DataFrame(
        {
            "cell_type": ["A", "A", "B", "B"],
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
        },
        index=["c1", "c2", "c3", "c4"],
    )
    Y = np.full((4, 3), 2.0, dtype=float)
    radii = [10.0, 20.0, 30.0]
    shared_default = prepare_shared_components(
        response_matrix=Y,
        metadata_df=meta,
        radius_values=radii,
        cell_type_col="cell_type",
        radius_mode="poly",
        poly_degree=2,
    )
    assert shared_default["response_normalization_factor"] is None
    assert np.allclose(shared_default["Y"], Y)

    shared_fixed = prepare_shared_components(
        response_matrix=Y,
        metadata_df=meta,
        radius_values=radii,
        cell_type_col="cell_type",
        radius_mode="poly",
        poly_degree=2,
        normalize_by=5.0,
    )
    assert np.isclose(shared_fixed["response_normalization_factor"], 5.0)
    assert np.allclose(shared_fixed["Y"], Y / 5.0)

    shared_raw = prepare_shared_components(
        response_matrix=Y,
        metadata_df=meta,
        radius_values=radii,
        cell_type_col="cell_type",
        radius_mode="poly",
        poly_degree=2,
        normalize_by_global_entropy=False,
    )
    assert shared_raw["response_normalization_factor"] is None
    assert np.allclose(shared_raw["Y"], Y)


def test_slide_level_cell_type_radius_model() -> None:
    pytest.importorskip("statsmodels")

    coords, labels, meta = _make_small_dataset()
    ld = compute_local_diversity_multi_radius(coords, labels, radii=[10, 20, 30, 40])

    shared = prepare_shared_components(
        response_matrix=ld.values,
        metadata_df=meta.loc[ld.index],
        radius_values=[10, 20, 30, 40],
        cell_type_col="cell_type",
        radius_mode="spline",
        n_radius_knots=4,
        spline_degree=2,
    )
    fit = fit_slide_level_cell_type_radius_model(shared)
    terms_df = summarize_model_terms(fit)
    effects_df = summarize_slide_level_cell_type_effects(fit, shared)

    assert {"term", "beta", "se", "pval", "t"}.issubset(terms_df.columns)
    assert {"cell_type", "beta_cell_type", "se_cell_type", "pval_cell_type", "t_cell_type"}.issubset(
        effects_df.columns
    )
    assert shared["reference_cell_type"] in effects_df["cell_type"].values
