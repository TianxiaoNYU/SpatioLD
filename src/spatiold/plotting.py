"""Visualization helpers for the SlideTag-style local-diversity workflow."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _get_plot_libs():
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError("matplotlib and seaborn are required for plotting functions.") from exc
    return plt, sns


def plot_spatial_cell_types(
    metadata_df: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    cell_type_col: str = "cell_type",
    palette: str = "tab20",
    point_size: float = 4,
    ax=None,
):
    """Scatter-plot cells in space colored by cell type."""
    plt, sns = _get_plot_libs()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6), dpi=200)

    sns.scatterplot(
        data=metadata_df,
        x=x_col,
        y=y_col,
        hue=cell_type_col,
        palette=palette,
        s=point_size,
        linewidth=0,
        ax=ax,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / microns")
    ax.set_ylabel("Y / microns")
    ax.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    return ax


def plot_mean_diversity_by_cell_type(
    summary_df: pd.DataFrame,
    *,
    radius_col: str = "radius",
    mean_col: str = "mean",
    std_col: str = "std",
    cell_type_col: str = "cell_type",
    title: str = "Mean Local Diversity vs Radius for Each Cell Type",
    ylabel: str = "Mean Local Diversity",
    ax=None,
):
    """Errorbar plot of mean local diversity by cell type across radii."""
    plt, _ = _get_plot_libs()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5), dpi=250)

    for ct, sub in summary_df.groupby(cell_type_col):
        sub = sub.sort_values(radius_col)
        ax.errorbar(
            sub[radius_col],
            sub[mean_col],
            yerr=sub[std_col],
            label=str(ct),
            marker="o",
            capsize=4,
            alpha=0.75,
        )

    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_xlabel("Neighborhood Radius (um)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    return ax


def plot_sample_vs_null_curve(
    summary_df: pd.DataFrame,
    *,
    radius_col: str = "radius",
    sample_mean_col: str = "sample_mean",
    sample_std_col: str = "sample_std",
    null_mean_col: str = "null_mean",
    null_ci_low_col: str = "null_ci_low",
    null_ci_high_col: str = "null_ci_high",
    ax=None,
):
    """Plot sample mean local diversity versus permutation-null baseline."""
    plt, _ = _get_plot_libs()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.5), dpi=250)

    s = summary_df.sort_values(radius_col)
    x = s[radius_col].to_numpy(dtype=float)

    ax.errorbar(
        x,
        s[sample_mean_col],
        # yerr=s[sample_std_col],
        fmt="o-",
        color="#1f77b4",
        # ecolor="#9ecae1",
        capsize=4,
        linewidth=2,
        label="Sample",
    )

    y_null = s[null_mean_col].to_numpy(dtype=float)
    yerr_null = np.vstack(
        [
            y_null - s[null_ci_low_col].to_numpy(dtype=float),
            s[null_ci_high_col].to_numpy(dtype=float) - y_null,
        ]
    )
    ax.errorbar(
        x,
        y_null,
        yerr=yerr_null,
        fmt="o-",
        color="gray",
        ecolor="lightgray",
        capsize=4,
        linewidth=2,
        label="Shuffled",
    )

    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.set_xlabel("Neighborhood Radius (um)")
    ax.set_ylabel("Mean Local Diversity")
    ax.set_title("Sample vs Null Local Diversity")
    ax.legend()
    return ax


def plot_gene_effect_volcano(
    results_df: pd.DataFrame,
    *,
    effect_col: str = "beta_gene",
    pval_col: str = "pval_gene",
    alpha: float = 0.05,
    ax=None,
):
    """Plot volcano-style scatter of gene effect versus significance."""
    plt, _ = _get_plot_libs()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5), dpi=260)

    pvals = np.clip(results_df[pval_col].to_numpy(dtype=float), 1e-300, 1.0)
    y = -np.log10(pvals)

    ax.scatter(
        results_df[effect_col],
        y,
        color="#1f77b4",
        edgecolor="black",
        linewidths=0.4,
        alpha=0.75,
    )
    ax.axhline(-np.log10(alpha), color="red", linestyle="--", label=f"p={alpha}")
    ax.set_xlabel("Gene Effect Coefficient")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Volcano Plot of Gene Effects")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    return ax


def plot_gene_set_venn(
    hvg_genes: Sequence[str],
    svg_genes: Sequence[str],
    ldvg_genes: Sequence[str],
    *,
    labels: tuple[str, str, str] = (
        "Highly Variable",
        "Spatially Variable",
        "Local-Diversity Variable",
    ),
    ax=None,
):
    """Plot 3-way Venn diagram of HVG/SVG/LDVG sets."""
    plt, _ = _get_plot_libs()
    try:
        from matplotlib_venn import venn3
    except ImportError as exc:
        raise ImportError("matplotlib-venn is required for `plot_gene_set_venn`.") from exc

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6), dpi=220)

    venn3([set(hvg_genes), set(svg_genes), set(ldvg_genes)], set_labels=labels, ax=ax)
    ax.set_title("Venn Diagram of HVG, SVG, and LDVG")
    return ax
