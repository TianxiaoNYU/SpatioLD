from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spatiold import cli


class _DummyFit:
    rsquared = 0.5
    rsquared_adj = 0.4
    aic = 1.0
    bic = 2.0


def _make_args(
    *,
    metadata_path: Path,
    expression_path: Path,
    output_dir: Path,
    simplify: bool,
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
        cell_size_col=None,
        min_fraction_expressed=0.0,
        min_genes_per_cell=0,
        n_top_hvg=3,
        hvg_flavor="seurat",
        simplify=simplify,
        cluster_method="scanpy-leiden",
        cluster_n_clusters=2,
        cluster_resolution=2.0,
        cluster_n_neighbors=5,
        cluster_n_pcs=3,
        random_state=0,
        radius_mode="poly",
        poly_degree=2,
        n_radius_knots=4,
        spline_degree=3,
        regression_normalize_by=None,
        no_regression_entropy_normalize=True,
        no_cluster_robust=True,
        quiet=True,
    )


def _write_small_inputs(tmp_path: Path) -> tuple[Path, Path, pd.Series]:
    cell_ids = [f"c{i}" for i in range(6)]
    meta = pd.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0, 2.0, 2.5],
            "y": [0.0, 0.0, 1.0, 1.0, 0.5, 1.5],
        },
        index=cell_ids,
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
    labels = pd.Series(
        ["0", "0", "0", "1", "1", "1"],
        index=cell_ids,
        name="cluster_label",
    )

    metadata_path = tmp_path / "metadata.csv"
    expression_path = tmp_path / "expression.csv"
    meta.to_csv(metadata_path)
    expr.to_csv(expression_path)
    return metadata_path, expression_path, labels


@pytest.mark.parametrize("simplify,expected_cluster_calls", [(False, 3), (True, 1)])
def test_cluster_pipeline_simplify_switches_clustering_frequency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    simplify: bool,
    expected_cluster_calls: int,
) -> None:
    metadata_path, expression_path, fixed_labels = _write_small_inputs(tmp_path)
    output_dir = tmp_path / ("out_simplify" if simplify else "out_default")
    output_dir.mkdir(parents=True, exist_ok=True)

    args = _make_args(
        metadata_path=metadata_path,
        expression_path=expression_path,
        output_dir=output_dir,
        simplify=simplify,
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

    call_counter = {"n": 0}

    def _fake_select_hvg(
        expr_df: pd.DataFrame,
        *,
        n_top_hvg: int,
        hvg_flavor: str,
        quiet: bool,
    ) -> pd.DataFrame:
        return hvg_df.head(n_top_hvg)

    def _fake_cluster(
        expr_df: pd.DataFrame,
        *,
        random_state: int,
        n_neighbors: int,
        n_pcs: int,
        resolution: float,
        n_clusters: int | None,
    ) -> pd.Series:
        del random_state, n_neighbors, n_pcs, resolution, n_clusters
        call_counter["n"] += 1
        return fixed_labels.loc[expr_df.index].copy()

    def _fake_fit_single_gene_radius_model(
        gene_values: np.ndarray | pd.Series,
        shared: dict[str, object],
        *,
        add_intercept: bool = True,
        cluster_robust: bool = True,
    ) -> dict[str, object]:
        del shared, add_intercept, cluster_robust
        beta = float(np.mean(np.asarray(gene_values, dtype=float)))
        return {
            "fit": _DummyFit(),
            "coef": pd.Series({"gene": beta}),
            "se": pd.Series({"gene": 1.0}),
            "pval": pd.Series({"gene": 0.5}),
            "feature_names": ["gene"],
        }

    monkeypatch.setattr(cli, "_select_hvg_with_fallback", _fake_select_hvg)
    monkeypatch.setattr(cli, "_cluster_cells_scanpy_leiden", _fake_cluster)
    monkeypatch.setattr(cli, "fit_single_gene_radius_model", _fake_fit_single_gene_radius_model)

    cli.run_cluster_pipeline(args)

    assert call_counter["n"] == expected_cluster_calls

    results_df = pd.read_csv(output_dir / "cluster_gene_ld_model_results.csv")
    assert set(results_df["gene"]) == set(hvg_cols)
    assert (results_df["n_clusters"] == 2).all()

    cluster_meta = pd.read_csv(output_dir / "cluster_meta_by_gene.csv", index_col=0)
    assert list(cluster_meta.columns) == hvg_cols
    if simplify:
        for gene in hvg_cols:
            assert cluster_meta[gene].astype(str).equals(fixed_labels.loc[cluster_meta.index].astype(str))

    summary = json.loads((output_dir / "run_summary.json").read_text())
    assert summary["simplify"] is simplify
    assert summary["clustering_runs"] == expected_cluster_calls
