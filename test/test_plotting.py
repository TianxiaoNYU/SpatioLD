from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

import matplotlib

matplotlib.use("Agg")

from spatiold import (
    plot_gene_effect_volcano,
    plot_kmeans_spatial_maps,
    plot_mean_diversity_by_cell_type,
    plot_sample_vs_null_curve,
    plot_significant_diversity_maps,
    plot_spatial_cell_types,
)


def test_basic_plotting_helpers() -> None:
    meta = pd.DataFrame(
        {
            "x": [0, 1, 2, 0, 1, 2],
            "y": [0, 0, 0, 1, 1, 1],
            "cell_type": ["A", "A", "B", "B", "C", "C"],
        },
        index=[f"c{i}" for i in range(6)],
    )

    summary_ct = pd.DataFrame(
        {
            "cell_type": ["A", "A", "B", "B"],
            "radius": [10, 20, 10, 20],
            "mean": [0.2, 0.4, 0.3, 0.5],
            "std": [0.05, 0.06, 0.04, 0.05],
            "n": [2, 2, 2, 2],
        }
    )

    summary_null = pd.DataFrame(
        {
            "radius": [10, 20],
            "sample_mean": [0.3, 0.45],
            "sample_std": [0.05, 0.06],
            "null_mean": [0.25, 0.35],
            "null_ci_low": [0.2, 0.3],
            "null_ci_high": [0.3, 0.4],
        }
    )

    pvals = pd.DataFrame({10: [0.01, 0.2, 0.9, 0.03, 0.8, 0.5], 20: [0.02, 0.3, 0.7, 0.4, 0.9, 0.04]}, index=meta.index)
    labels_df = pd.DataFrame({"ld_kmeans_k2": ["0", "0", "1", "1", "1", "0"]}, index=meta.index)
    results_df = pd.DataFrame({"beta_gene": [0.2, -0.1, 0.5], "pval_gene": [0.01, 0.2, 1e-4]})

    ax1 = plot_spatial_cell_types(meta)
    ax2 = plot_mean_diversity_by_cell_type(summary_ct)
    ax3 = plot_sample_vs_null_curve(summary_null)
    fig4, _ = plot_kmeans_spatial_maps(meta, labels_df, k_values=[2], ncols=1)
    fig5, _ = plot_significant_diversity_maps(meta[["x", "y"]], pvals, ncols=2)
    ax6 = plot_gene_effect_volcano(results_df)

    assert ax1 is not None and ax2 is not None and ax3 is not None and ax6 is not None
    assert fig4 is not None and fig5 is not None
