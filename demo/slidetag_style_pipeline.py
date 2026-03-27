"""SlideTag-style end-to-end demo using example metadata + synthetic expression.

This demo reads a metadata CSV from `example_data/`, computes local-diversity
pipeline outputs, and generates synthetic expression (default 1000 genes)
for downstream gene-level modeling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import spatiold as sld


def _load_metadata(metadata_path: Path, max_cells: int | None, seed: int) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path)

    if "unique_id" in meta.columns:
        meta = meta.set_index("unique_id")
    else:
        meta.index = meta.index.astype(str)

    required_cols = {"x", "y", "cell_type"}
    missing = required_cols.difference(meta.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    meta = meta.loc[:, ["x", "y", "cell_type"] + [c for c in meta.columns if c not in {"x", "y", "cell_type"}]]
    meta = meta.dropna(subset=["x", "y", "cell_type"])
    meta.index = meta.index.astype(str)

    if max_cells is not None and len(meta) > max_cells:
        rng = np.random.default_rng(seed)
        picked = rng.choice(meta.index.to_numpy(), size=max_cells, replace=False)
        meta = meta.loc[picked].sort_index()

    return meta


def _synthetic_expression(meta: pd.DataFrame, n_genes: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_cells = len(meta)

    expr = rng.normal(size=(n_cells, n_genes)).astype(np.float32)
    expr_df = pd.DataFrame(
        expr,
        index=meta.index,
        columns=[f"gene_{i}" for i in range(n_genes)],
    )

    # Inject structured signal based on cell types for the first few genes.
    ct_codes = meta["cell_type"].astype("category").cat.codes.to_numpy(dtype=float)
    ct_codes = (ct_codes - ct_codes.mean()) / (ct_codes.std() + 1e-8)

    n_signal_genes = min(20, n_genes)
    for i in range(n_signal_genes):
        weight = 0.15 + 0.05 * (i % 5)
        expr_df.iloc[:, i] = expr_df.iloc[:, i] + weight * ct_codes

    return expr_df


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_metadata = root / "example_data" / "SlideTag_HumanCortex.csv"

    parser = argparse.ArgumentParser(description="Run SlideTag-style SpatioLD demo pipeline.")
    parser.add_argument("--metadata", type=Path, default=default_metadata, help="Path to metadata CSV.")
    parser.add_argument("--max-cells", type=int, default=2500, help="Subsample metadata to this many cells for runtime control.")
    parser.add_argument("--n-genes", type=int, default=1000, help="Number of synthetic expression genes to generate.")
    parser.add_argument("--n-model-genes", type=int, default=1000, help="Number of genes used in regression/SVG steps.")
    parser.add_argument("--n-perm", type=int, default=50, help="Permutations for p-values and null distribution.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--plot", action="store_true", help="Generate plots (off by default).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")

    meta = _load_metadata(args.metadata, args.max_cells, args.seed)

    n_genes = int(max(1, args.n_genes))
    n_model_genes = int(max(1, min(args.n_model_genes, n_genes)))

    expr_df = _synthetic_expression(meta, n_genes=n_genes, seed=args.seed)
    adata = ad.AnnData(
        X=expr_df.values,
        obs=meta.copy(),
        var=pd.DataFrame(index=expr_df.columns.astype(str)),
    )
    adata.obs_names = expr_df.index.astype(str)
    obj = sld.SpatioLD.from_anndata(
        adata,
        label_key="cell_type",
        coord_keys=("x", "y"),
    )
    expr_model_df = obj.to_anndata().to_df().iloc[:, :n_model_genes].copy()

    radii = [30, 60, 90, 120, 150, 180, 210, 240, 270]

    # ------------------------------------------------------------------
    # 1) Local diversity + permutation
    # ------------------------------------------------------------------
    ld_df = obj.compute_local_diversity(radii=radii, key="ld_full")
    perm_stats = obj.compute_permutation_stats(
        n_perm=args.n_perm,
        radii=radii,
        random_state=args.seed,
        n_jobs=1,
        pvals_key="ld_pvals",
        perm_mean_key="ld_perm_mean",
    )
    pvals_df = perm_stats["pvals"]
    perm_dist = perm_stats["distribution"]

    entropy_global = obj.compute_global_shannon_entropy()
    summary_ct = obj.summarize_local_diversity_by_cell_type(
        local_diversity_key="ld_full",
    )
    summary_null = obj.compute_sample_vs_null_summary(
        perm_dist,
        local_diversity_key="ld_full",
    )

    # ------------------------------------------------------------------
    # 2) Clustering + significance map
    # ------------------------------------------------------------------
    cluster_labels_df, _ = obj.cluster_local_diversity_profiles(
        local_diversity_key="ld_full",
        k_values=(2, 3, 4),
    )
    sig_mask_df = obj.build_significance_mask(pvals_key="ld_pvals", alpha=0.05)

    # ------------------------------------------------------------------
    # 3) Gene-radius model + SVG
    # ------------------------------------------------------------------
    shared = obj.prepare_shared_components(
        local_diversity_key="ld_full",
        radius_mode="poly",
        poly_degree=3,
        normalize_by_global_entropy=True,
    )
    results_df, fit_objects = sld.fit_all_genes(
        expr_model_df,
        shared,
        cluster_robust=True,
        verbose=False,
    )
    svg_df = obj.compute_svg_morans_i(expr_model_df, k=15)

    # ------------------------------------------------------------------
    # 4) Optional plotting
    # ------------------------------------------------------------------
    if args.plot:
        import matplotlib.pyplot as plt

        ax1 = sld.plot_spatial_cell_types(obj.metadata)
        ax2 = sld.plot_mean_diversity_by_cell_type(
            summary_ct,
            title="Normalized Mean Local Diversity vs Radius by Cell Type",
            ylabel="Mean Local Diversity (normalized)",
        )
        ax3 = sld.plot_sample_vs_null_curve(summary_null)
        fig4, _ = sld.plot_kmeans_spatial_maps(obj.metadata, cluster_labels_df, k_values=[2, 3, 4], ncols=2)
        fig5, _ = sld.plot_significant_diversity_maps(obj.get_coords_df(), pvals_df, alpha=0.05)
        ax6 = sld.plot_gene_effect_volcano(results_df)

        try:
            hvg_mock = set(results_df.head(30)["gene"])
            svg_top = set(svg_df.head(30)["gene"])
            ldvg_top = set(results_df.nsmallest(30, "pval_gene")["gene"])
            ax7 = sld.plot_gene_set_venn(hvg_mock, svg_top, ldvg_top)  # type: ignore
        except ImportError:
            ax7 = None

        for ax in [ax1, ax2, ax3, ax6, ax7]:
            if ax is not None:
                ax.figure.tight_layout()
        for fig in [fig4, fig5]:
            fig.tight_layout()

        plt.close("all")

    print("SlideTag-style pipeline demo complete.")
    print(f"Metadata file: {args.metadata}")
    print(f"Cells used: {len(meta)}")
    print(f"Synthetic expression shape: {expr_df.shape} (modeled genes: {n_model_genes})")
    print(f"Global Shannon entropy: {entropy_global:.4f}")
    print(f"LD matrix shape: {ld_df.shape}")
    print(f"Regression summary rows: {results_df.shape[0]}")
    print(f"Top 5 genes by p-value: {results_df.head(5)['gene'].tolist()}")
    print(f"Top 5 SVG genes by Moran's I: {svg_df.head(5)['gene'].tolist()}")
    print(f"Significance mask shape: {sig_mask_df.shape}")
    print(f"Stored fit objects: {len(fit_objects)}")


if __name__ == "__main__":
    main()
