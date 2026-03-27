from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatiold import (
    cluster_local_diversity_profiles,
    compute_global_shannon_entropy,
    compute_local_diversity_multi_radius,
    compute_nd_permutation_distribution,
    compute_sample_vs_null_summary,
    compute_svg_morans_i,
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
    return coords, labels, meta


def test_pipeline_summaries_and_clustering() -> None:
    coords, labels, meta = _make_small_dataset()
    ld = compute_local_diversity_multi_radius(coords, labels, radii=[10, 20, 30])

    entropy = compute_global_shannon_entropy(labels)
    summary_ct = summarize_local_diversity_by_cell_type(ld, meta, normalize_by=entropy)
    assert {"cell_type", "radius", "mean", "std", "n"}.issubset(summary_ct.columns)

    dist = compute_nd_permutation_distribution(coords, labels, n_perm=8, radii=[10, 20, 30], n_jobs=1)
    summary_null = compute_sample_vs_null_summary(ld, dist, normalize_by=entropy)
    assert {"radius", "sample_mean", "null_mean", "null_ci_low", "null_ci_high"}.issubset(summary_null.columns)

    labels_df, models = cluster_local_diversity_profiles(ld, k_values=(2, 3))
    assert labels_df.shape[0] == ld.shape[0]
    assert set(labels_df.columns) == {"ld_kmeans_k2", "ld_kmeans_k3"}
    assert set(models.keys()) == {2, 3}


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
    )

    fit = fit_single_gene_radius_model(expr["g0"].values, shared)
    assert "gene" in fit["coef"].index

    svg_df = compute_svg_morans_i(expr, coords)
    assert list(svg_df.columns) == ["gene", "moran_I"]
    assert svg_df.shape[0] == expr.shape[1]


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
    entropy = compute_global_shannon_entropy(meta["cell_type"])

    shared_default = prepare_shared_components(
        response_matrix=Y,
        metadata_df=meta,
        radius_values=radii,
        cell_type_col="cell_type",
        radius_mode="poly",
        poly_degree=2,
    )
    assert np.isclose(shared_default["response_normalization_factor"], entropy)
    assert np.allclose(shared_default["Y"], Y / entropy)

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
