from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spatiold import cli


def _make_args(
    *,
    metadata_path: Path,
    expression_path: Path,
    output_dir: Path,
    gene_model_mode: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        metadata=metadata_path,
        expression=expression_path,
        input_h5ad=None,
        h5ad_layer=None,
        spatial_key="spatial",
        output_dir=output_dir,
        radii=["20", "40"],
        cell_id_col=None,
        x_col="x",
        y_col="y",
        cell_type_col="cell_type",
        cell_size_col=None,
        min_fraction_expressed=0.0,
        min_genes_per_cell=0,
        n_model_genes=3,
        k_values=[2],
        svg_k=3,
        radius_mode="poly",
        poly_degree=2,
        n_radius_knots=4,
        spline_degree=3,
        gene_model_mode=gene_model_mode,
        regression_normalize_by=None,
        no_regression_entropy_normalize=True,
        no_cluster_robust=True,
        quiet=True,
        n_perm=5,
        pval_pooling="neighborhood-size",
        random_state=0,
        alpha=0.05,
        save_permutation_distribution=False,
    )


def _write_small_inputs(tmp_path: Path) -> tuple[Path, Path]:
    cell_ids = [f"c{i}" for i in range(6)]
    meta = pd.DataFrame(
        {
            "unique_id": cell_ids,
            "x": [0.0, 1.0, 0.0, 1.0, 2.0, 2.5],
            "y": [0.0, 0.0, 1.0, 1.0, 0.5, 1.5],
            "cell_type": ["A", "A", "B", "B", "A", "B"],
        }
    )
    expr = pd.DataFrame(
        np.array(
            [
                [5, 1, 2, 3],
                [4, 2, 1, 3],
                [5, 2, 2, 4],
                [1, 4, 5, 2],
                [2, 5, 4, 1],
                [1, 3, 5, 2],
            ],
            dtype=float,
        ),
        index=cell_ids,
        columns=["g0", "g1", "g2", "g3"],
    )

    metadata_path = tmp_path / "metadata.csv"
    expression_path = tmp_path / "expression.csv"
    meta.to_csv(metadata_path, index=False)
    expr.to_csv(expression_path)
    return metadata_path, expression_path


@pytest.mark.parametrize("gene_model_mode", ["single", "joint"])
def test_run_pipeline_gene_model_switch_dispatches_correct_fitter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gene_model_mode: str,
) -> None:
    metadata_path, expression_path = _write_small_inputs(tmp_path)
    output_dir = tmp_path / f"out_{gene_model_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    args = _make_args(
        metadata_path=metadata_path,
        expression_path=expression_path,
        output_dir=output_dir,
        gene_model_mode=gene_model_mode,
    )

    hvg_cols = ["g0", "g1", "g2"]
    hvg_df = pd.DataFrame(
        {
            "means": [1.0, 1.1, 1.2],
            "dispersions": [2.0, 1.9, 1.8],
            "dispersions_norm": [2.0, 1.9, 1.8],
        },
        index=hvg_cols,
    )

    calls = {"single": 0, "joint": 0}

    def _fake_select_hvg(
        expr_df: pd.DataFrame,
        *,
        n_top_hvg: int,
        hvg_flavor: str,
        quiet: bool,
    ) -> pd.DataFrame:
        del expr_df, hvg_flavor, quiet
        return hvg_df.head(n_top_hvg)

    def _fake_fit_all_genes(
        expr_df: pd.DataFrame,
        shared: dict[str, object],
        *,
        cluster_robust: bool,
        verbose: bool,
        store_fit_objects: bool,
        method: str = "auto",
        chunk_size: int = 64,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
        del shared, cluster_robust, verbose, store_fit_objects, method, chunk_size
        calls["single"] += 1
        return (
            pd.DataFrame(
                {
                    "gene": expr_df.columns.astype(str),
                    "beta_gene": np.arange(expr_df.shape[1], dtype=float),
                    "se_gene": 1.0,
                    "pval_gene": 0.5,
                    "t_gene": 0.0,
                    "r2": 0.1,
                    "adj_r2": 0.05,
                    "aic": 1.0,
                    "bic": 2.0,
                }
            ),
            {},
        )

    def _fake_fit_joint_gene_radius_model(
        expr_df: pd.DataFrame,
        shared: dict[str, object],
        *,
        add_intercept: bool = True,
        cluster_robust: bool = True,
        return_fit_object: bool = True,
    ) -> dict[str, object]:
        del shared, add_intercept, cluster_robust, return_fit_object
        calls["joint"] += 1
        feature_names = ["const", *expr_df.columns.astype(str), "cell_type_B", "poly_radius_deg1"]
        coef = pd.Series([0.1, 1.0, -0.5, 0.2, 0.3, -0.1], index=feature_names)
        se = pd.Series([1.0] * len(feature_names), index=feature_names)
        pval = pd.Series([0.9, 0.01, 0.02, 0.03, 0.5, 0.4], index=feature_names)
        gene_summary = pd.DataFrame(
            {
                "gene": expr_df.columns.astype(str),
                "beta_gene": [1.0, -0.5, 0.2],
                "se_gene": [1.0, 1.0, 1.0],
                "pval_gene": [0.01, 0.02, 0.03],
                "t_gene": [1.0, -0.5, 0.2],
            }
        )
        return {
            "coef": coef,
            "se": se,
            "pval": pval,
            "feature_names": feature_names,
            "gene_summary": gene_summary,
        }

    monkeypatch.setattr(cli, "_select_hvg_with_fallback", _fake_select_hvg)
    monkeypatch.setattr(cli, "fit_all_genes", _fake_fit_all_genes)
    monkeypatch.setattr(cli, "fit_joint_gene_radius_model", _fake_fit_joint_gene_radius_model)

    cli.run_pipeline(args, skip_permutation=True)

    assert calls["single"] == (1 if gene_model_mode == "single" else 0)
    assert calls["joint"] == (1 if gene_model_mode == "joint" else 0)

    results_df = pd.read_csv(output_dir / "gene_radius_model_results.csv")
    assert set(results_df["gene"]) == set(hvg_cols)

    terms_path = output_dir / "gene_radius_model_terms.csv"
    assert terms_path.exists() == (gene_model_mode == "joint")
    if gene_model_mode == "joint":
        terms_df = pd.read_csv(terms_path)
        assert "term" in terms_df.columns
        assert set(hvg_cols).issubset(set(terms_df["term"]))

    summary = json.loads((output_dir / "run_summary.json").read_text())
    assert summary["gene_model_mode"] == gene_model_mode
