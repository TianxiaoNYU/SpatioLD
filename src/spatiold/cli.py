"""Command-line pipeline runner for SpatioLD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .core import SpatioLD
from .diversity import compute_local_diversity_multi_radius
from .pipeline import (
    align_expression_and_metadata,
    compute_hvg_scanpy,
    fit_all_genes,
    fit_single_gene_radius_model,
    fit_slide_level_cell_type_radius_model,
    prepare_shared_components,
    preprocess_expression_matrix,
    summarize_model_terms,
    summarize_slide_level_cell_type_effects,
)

DEFAULT_RADII = [30, 60, 90, 120, 150, 180, 210, 240, 270]


def _parse_radii(tokens: list[str]) -> list[float]:
    out: list[float] = []
    for token in tokens:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(float(part))
    if not out:
        raise ValueError("At least one radius must be provided.")
    return out


def _read_table(path: Path, *, index_col: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, index_col=index_col)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", index_col=index_col)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {suffix}. Use .csv, .tsv/.txt, or .parquet.")


def _load_metadata(
    metadata_path: Path,
    *,
    cell_id_col: str | None,
    x_col: str,
    y_col: str,
    cell_type_col: str | None,
) -> pd.DataFrame:
    meta = _read_table(metadata_path, index_col=None)

    if cell_id_col is not None:
        if cell_id_col not in meta.columns:
            raise KeyError(f"`{cell_id_col}` not found in metadata columns.")
        meta = meta.set_index(cell_id_col)
    elif "unique_id" in meta.columns:
        meta = meta.set_index("unique_id")
    elif meta.index.name is None:
        meta.index = meta.index.astype(str)

    required = [x_col, y_col]
    if cell_type_col is not None:
        required.append(cell_type_col)
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise KeyError(f"Metadata missing required columns: {missing}")

    meta = meta.dropna(subset=required).copy()
    meta.index = meta.index.astype(str)
    return meta


def _load_expression(expr_path: Path) -> pd.DataFrame:
    expr_raw = _read_table(expr_path, index_col=0)
    expr_raw.index = expr_raw.index.astype(str)
    expr_raw.columns = expr_raw.columns.astype(str)
    return expr_raw


def _orient_expression(expr_raw: pd.DataFrame, meta_index: pd.Index) -> pd.DataFrame:
    meta_ids = set(meta_index.astype(str))
    overlap_index = len(set(expr_raw.index).intersection(meta_ids))
    overlap_cols = len(set(expr_raw.columns).intersection(meta_ids))
    if overlap_cols > overlap_index:
        expr = expr_raw.T
    else:
        expr = expr_raw
    expr.index = expr.index.astype(str)
    expr.columns = expr.columns.astype(str)
    return expr


def _write_args_snapshot(
    args: argparse.Namespace,
    output_dir: Path,
    radii: list[float],
    *,
    permutation_enabled: bool,
) -> None:
    payload = vars(args).copy()
    payload["radii"] = radii
    payload["metadata"] = str(args.metadata)
    payload["expression"] = str(args.expression)
    payload["output_dir"] = str(args.output_dir)
    payload["permutation_enabled"] = permutation_enabled
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2))


def _add_common_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata path (.csv/.tsv/.txt/.parquet).")
    parser.add_argument("--expression", type=Path, required=True, help="Expression matrix path (.csv/.tsv/.txt/.parquet).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write pipeline outputs.")

    parser.add_argument(
        "--radii",
        nargs="+",
        default=[str(r) for r in DEFAULT_RADII],
        help="Radii as space-separated values (e.g. --radii 30 60 90) or comma list (e.g. --radii 30,60,90).",
    )

    parser.add_argument("--cell-id-col", type=str, default=None, help="Metadata column containing cell IDs. If omitted, use `unique_id` when present, otherwise existing index.")
    parser.add_argument("--x-col", type=str, default="x", help="Metadata x-coordinate column.")
    parser.add_argument("--y-col", type=str, default="y", help="Metadata y-coordinate column.")
    parser.add_argument("--cell-type-col", type=str, default="cell_type", help="Metadata cell-type label column.")

    parser.add_argument("--min-fraction-expressed", type=float, default=0.02, help="Gene filter: minimum fraction of cells with nonzero expression.")
    parser.add_argument("--min-genes-per-cell", type=int, default=50, help="Cell filter: minimum nonzero genes per cell.")
    parser.add_argument("--n-model-genes", type=int, default=1000, help="Top variable genes used for regression and SVG ranking.")

    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 3, 4, 5], help="K values for local-diversity profile clustering.")
    parser.add_argument("--svg-k", type=int, default=15, help="kNN size for Moran's I SVG score.")

    parser.add_argument("--radius-mode", type=str, choices=["spline", "poly"], default="spline", help="Radius basis mode for gene model.")
    parser.add_argument("--poly-degree", type=int, default=3, help="Polynomial degree when `--radius-mode poly`.")
    parser.add_argument("--n-radius-knots", type=int, default=5, help="Spline knots when `--radius-mode spline`.")
    parser.add_argument("--spline-degree", type=int, default=3, help="Spline degree when `--radius-mode spline`.")
    parser.add_argument(
        "--regression-normalize-by",
        type=float,
        default=None,
        help="Optional fixed divisor for local-diversity response before gene modeling. If omitted, global entropy normalization is used.",
    )
    parser.add_argument(
        "--no-regression-entropy-normalize",
        action="store_true",
        help="Disable default global-entropy normalization of local-diversity response in gene modeling.",
    )
    parser.add_argument("--no-cluster-robust", action="store_true", help="Disable cluster-robust standard errors in gene model.")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress messages.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run full SpatioLD pipeline (including permutation inference)."
    )
    _add_common_pipeline_arguments(parser)
    parser.add_argument("--n-perm", type=int, default=100, help="Permutation count for p-values and null summaries.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance alpha for p-value mask.")
    parser.add_argument("--save-permutation-distribution", action="store_true", help="Also save full permutation distribution to `.npz`.")
    return parser


def build_slim_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SpatioLD pipeline in slim mode (skip permutation inference; keep all non-permutation steps)."
        )
    )
    _add_common_pipeline_arguments(parser)
    return parser


def build_cluster_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run leave-one-gene-out cluster-label SpatioLD workflow (no cell-type labels required)."
        )
    )
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata path (.csv/.tsv/.txt/.parquet) with coordinates.")
    parser.add_argument("--expression", type=Path, required=True, help="Expression matrix path (.csv/.tsv/.txt/.parquet).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write workflow outputs.")
    parser.add_argument(
        "--radii",
        nargs="+",
        default=[str(r) for r in DEFAULT_RADII],
        help="Radii as space-separated values (e.g. --radii 30 60 90) or comma list (e.g. --radii 30,60,90).",
    )
    parser.add_argument("--cell-id-col", type=str, default=None, help="Metadata column containing cell IDs. If omitted, use `unique_id` when present, otherwise existing index.")
    parser.add_argument("--x-col", type=str, default="x", help="Metadata x-coordinate column.")
    parser.add_argument("--y-col", type=str, default="y", help="Metadata y-coordinate column.")

    parser.add_argument("--min-fraction-expressed", type=float, default=0.02, help="Gene filter: minimum fraction of cells with nonzero expression.")
    parser.add_argument("--min-genes-per-cell", type=int, default=50, help="Cell filter: minimum nonzero genes per cell.")
    parser.add_argument("--n-top-hvg", type=int, default=100, help="Top HVGs used for leave-one-gene-out modeling.")
    parser.add_argument("--hvg-flavor", type=str, default="seurat", help="Scanpy HVG flavor (default: seurat).")

    parser.add_argument("--cluster-method", type=str, choices=["scanpy-leiden"], default="scanpy-leiden", help="Clustering backend (default: scanpy-leiden).")
    parser.add_argument(
        "--cluster-n-clusters",
        type=int,
        default=None,
        help=(
            "Optional direct number of clusters. "
            "If provided, this takes priority over `--cluster-resolution`."
        ),
    )
    parser.add_argument("--cluster-resolution", type=float, default=2.0, help="Leiden resolution. Keep default for now; exposed for future tuning.")
    parser.add_argument("--cluster-n-neighbors", type=int, default=15, help="Neighbors for clustering graph construction.")
    parser.add_argument("--cluster-n-pcs", type=int, default=30, help="PCA components for clustering graph construction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for clustering.")

    parser.add_argument("--radius-mode", type=str, choices=["spline", "poly"], default="spline", help="Radius basis mode for gene model.")
    parser.add_argument("--poly-degree", type=int, default=3, help="Polynomial degree when `--radius-mode poly`.")
    parser.add_argument("--n-radius-knots", type=int, default=5, help="Spline knots when `--radius-mode spline`.")
    parser.add_argument("--spline-degree", type=int, default=3, help="Spline degree when `--radius-mode spline`.")
    parser.add_argument(
        "--regression-normalize-by",
        type=float,
        default=None,
        help="Optional fixed divisor for local-diversity response before modeling. If omitted, global entropy normalization is used.",
    )
    parser.add_argument(
        "--no-regression-entropy-normalize",
        action="store_true",
        help="Disable default global-entropy normalization of local-diversity response in modeling.",
    )
    parser.add_argument("--no-cluster-robust", action="store_true", help="Disable cluster-robust standard errors in gene model.")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress messages.")
    return parser


def _build_sample_only_null_summary(local_diversity_df: pd.DataFrame) -> pd.DataFrame:
    obs = local_diversity_df.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "radius": [float(c) for c in local_diversity_df.columns],
            "sample_mean": obs.mean(axis=0),
            "sample_std": obs.std(axis=0),
            "null_mean": np.nan,
            "null_ci_low": np.nan,
            "null_ci_high": np.nan,
        }
    )


def _cluster_cells_scanpy_leiden(
    expr_df: pd.DataFrame,
    *,
    random_state: int,
    n_neighbors: int,
    n_pcs: int,
    resolution: float,
    n_clusters: int | None,
) -> pd.Series:
    if n_clusters is not None and int(n_clusters) < 2:
        raise ValueError("`n_clusters` must be >= 2 when provided.")

    try:
        import scanpy as sc
    except ImportError:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        X = np.log1p(expr_df.to_numpy(dtype=float))
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
        n_clusters_use = (
            int(n_clusters)
            if n_clusters is not None
            else int(max(2, min(expr_df.shape[0], round(max(2.0, 4.0 * float(resolution))))))
        )
        km = KMeans(n_clusters=n_clusters_use, random_state=random_state, n_init=20)
        labels = pd.Series(km.fit_predict(X).astype(str), index=expr_df.index.astype(str), name="cluster_label")
        return labels

    if expr_df.shape[0] < 2:
        raise ValueError("Need at least 2 cells for clustering.")

    adata = ad.AnnData(X=expr_df.to_numpy(dtype=np.float32))
    adata.obs_names = expr_df.index.astype(str)
    adata.var_names = expr_df.columns.astype(str)

    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)
    ## Skip normalization/logging since we're only using this for clustering, not modeling. 
    ## Also align with the non-normalized expression used for regression in the cluster-label workflow.

    n_neighbors_use = int(max(1, min(n_neighbors, adata.n_obs - 1)))
    max_pcs = int(min(adata.n_obs - 1, adata.n_vars - 1))
    if max_pcs >= 1:
        n_pcs_use = int(max(1, min(n_pcs, max_pcs)))
        sc.tl.pca(adata, n_comps=n_pcs_use, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors_use, n_pcs=n_pcs_use)
    else:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors_use, use_rep="X")

    # Priority rule: if explicit cluster count is given, use it directly.
    if n_clusters is not None:
        from sklearn.cluster import KMeans

        X_cluster = adata.obsm["X_pca"] if "X_pca" in adata.obsm else np.asarray(adata.X)
        n_clusters_use = int(min(max(2, int(n_clusters)), adata.n_obs))
        km = KMeans(n_clusters=n_clusters_use, random_state=random_state, n_init=20)
        labels = pd.Series(km.fit_predict(X_cluster).astype(str), index=adata.obs_names, name="cluster_label")
        labels.index = labels.index.astype(str)
        return labels

    try:
        sc.tl.leiden(adata, resolution=float(resolution), key_added="cluster_label", random_state=random_state)
        labels = adata.obs["cluster_label"].astype(str)
    except Exception:
        from sklearn.cluster import KMeans

        X_cluster = adata.obsm["X_pca"] if "X_pca" in adata.obsm else np.asarray(adata.X)
        n_clusters = int(max(2, min(adata.n_obs, round(max(2.0, 4.0 * float(resolution))))))
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        labels = pd.Series(km.fit_predict(X_cluster).astype(str), index=adata.obs_names, name="cluster_label")

    labels.index = labels.index.astype(str)
    return labels


def _select_hvg_with_fallback(
    expr_df: pd.DataFrame,
    *,
    n_top_hvg: int,
    hvg_flavor: str,
    quiet: bool,
) -> pd.DataFrame:
    try:
        return compute_hvg_scanpy(
            expr_df,
            n_top_genes=n_top_hvg,
            flavor=hvg_flavor,
        )
    except ImportError:
        if not quiet:
            print("scanpy not available; falling back to variance-ranked HVGs.")

        mean_s = expr_df.mean(axis=0)
        var_s = expr_df.var(axis=0, ddof=0)
        disp = var_s / np.maximum(mean_s, 1e-12)
        hvg_df = pd.DataFrame(
            {
                "means": mean_s,
                "dispersions": disp,
                "dispersions_norm": disp,
            }
        ).sort_values("dispersions_norm", ascending=False)
        return hvg_df.head(n_top_hvg)


def run_pipeline(args: argparse.Namespace, *, skip_permutation: bool = False) -> None:
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")
    if not args.expression.exists():
        raise FileNotFoundError(f"Expression file not found: {args.expression}")

    radii = _parse_radii(args.radii)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_raw = _load_metadata(
        args.metadata,
        cell_id_col=args.cell_id_col,
        x_col=args.x_col,
        y_col=args.y_col,
        cell_type_col=args.cell_type_col,
    )
    expr_raw = _load_expression(args.expression)
    expr = _orient_expression(expr_raw, meta_raw.index)

    expr_filt = preprocess_expression_matrix(
        expr,
        min_fraction_expressed=args.min_fraction_expressed,
        min_genes_per_cell=args.min_genes_per_cell,
    )
    expr_aligned, meta_aligned = align_expression_and_metadata(expr_filt, meta_raw)
    if expr_aligned.empty or meta_aligned.empty:
        raise ValueError("No overlapping cells found between metadata and expression after filtering/alignment.")

    meta_obj = meta_aligned.copy()
    rename_cols = {}
    if args.x_col != "x":
        rename_cols[args.x_col] = "x"
    if args.y_col != "y":
        rename_cols[args.y_col] = "y"
    if args.cell_type_col != "cell_type":
        rename_cols[args.cell_type_col] = "cell_type"
    if rename_cols:
        meta_obj = meta_obj.rename(columns=rename_cols)

    adata = ad.AnnData(
        X=expr_aligned.values,
        obs=meta_obj,
        var=pd.DataFrame(index=expr_aligned.columns.astype(str)),
    )
    adata.obs_names = expr_aligned.index.astype(str)

    obj = SpatioLD.from_anndata(
        adata,
        label_key="cell_type",
        coord_keys=("x", "y"),
    )

    ld_key = "spatiold_local_diversity"
    pval_key = "spatiold_local_diversity_pvals"
    perm_mean_key = "spatiold_local_diversity_perm_mean"

    ld_df = obj.compute_local_diversity(radii=radii, key=ld_key)
    summary_ct = obj.summarize_local_diversity_by_cell_type(local_diversity_key=ld_key)
    cluster_labels_df, _ = obj.cluster_local_diversity_profiles(local_diversity_key=ld_key, k_values=args.k_values)

    if skip_permutation:
        pvals_df = pd.DataFrame(np.nan, index=ld_df.index, columns=ld_df.columns)
        perm_mean_df = pd.DataFrame(np.nan, index=ld_df.index, columns=ld_df.columns)
        summary_null = _build_sample_only_null_summary(ld_df)
        sig_mask_df = pd.DataFrame(0, index=ld_df.index, columns=ld_df.columns, dtype=int)
        perm_dist = None
    else:
        perm_stats = obj.compute_permutation_stats(
            n_perm=args.n_perm,
            radii=radii,
            random_state=args.random_state,
            n_jobs=-1,
            store=True,
            pvals_key=pval_key,
            perm_mean_key=perm_mean_key,
        )
        pvals_df = perm_stats["pvals"]
        perm_mean_df = perm_stats["perm_mean"]
        perm_dist = perm_stats["distribution"]
        summary_null = obj.compute_sample_vs_null_summary(perm_dist, local_diversity_key=ld_key)
        sig_mask_df = obj.build_significance_mask(pvals_key=pval_key, alpha=args.alpha)

    shared = obj.prepare_shared_components(
        local_diversity_key=ld_key,
        radius_mode=args.radius_mode,
        poly_degree=args.poly_degree,
        n_radius_knots=args.n_radius_knots,
        spline_degree=args.spline_degree,
        normalize_by=args.regression_normalize_by,
        normalize_by_global_entropy=not args.no_regression_entropy_normalize,
    )
    slide_ct_fit = fit_slide_level_cell_type_radius_model(
        shared,
        cluster_robust=not args.no_cluster_robust,
    )
    slide_ct_terms_df = summarize_model_terms(slide_ct_fit)
    slide_ct_effects_df = summarize_slide_level_cell_type_effects(slide_ct_fit, shared)

    n_model_genes = max(1, min(int(args.n_model_genes), expr_aligned.shape[1]))
    var_rank = expr_aligned.var(axis=0).sort_values(ascending=False)
    model_genes = var_rank.head(n_model_genes).index
    expr_model = expr_aligned.loc[:, model_genes].copy()

    results_df, _ = fit_all_genes(
        expr_model,
        shared,
        cluster_robust=not args.no_cluster_robust,
        verbose=not args.quiet,
    )
    svg_df = obj.compute_svg_morans_i(expr_model, k=args.svg_k)

    ld_df.to_csv(output_dir / "local_diversity.csv")
    pvals_df.to_csv(output_dir / "local_diversity_pvals.csv")
    perm_mean_df.to_csv(output_dir / "local_diversity_perm_mean.csv")
    summary_ct.to_csv(output_dir / "summary_by_cell_type.csv", index=False)
    summary_null.to_csv(output_dir / "summary_sample_vs_null.csv", index=False)
    cluster_labels_df.to_csv(output_dir / "cluster_labels.csv")
    sig_mask_df.to_csv(output_dir / "significance_mask.csv")
    slide_ct_terms_df.to_csv(output_dir / "slide_cell_type_radius_model_terms.csv", index=False)
    slide_ct_effects_df.to_csv(output_dir / "slide_cell_type_effects.csv", index=False)
    results_df.to_csv(output_dir / "gene_radius_model_results.csv", index=False)
    svg_df.to_csv(output_dir / "svg_morans_i.csv", index=False)

    if (not skip_permutation) and getattr(args, "save_permutation_distribution", False):
        np.savez_compressed(output_dir / "permutation_distribution.npz", permutation_distribution=perm_dist)

    _write_args_snapshot(args, output_dir, radii, permutation_enabled=not skip_permutation)

    summary_payload = {
        "n_cells": int(expr_aligned.shape[0]),
        "n_genes_after_filter": int(expr_aligned.shape[1]),
        "n_model_genes": int(expr_model.shape[1]),
        "n_cell_types": int(len(shared["cell_type_levels"])),
        "reference_cell_type": str(shared["reference_cell_type"]),
        "response_normalization_factor": shared["response_normalization_factor"],
        "n_radii": int(len(radii)),
        "radii": radii,
        "permutation_enabled": not skip_permutation,
        "n_perm": int(args.n_perm) if not skip_permutation else 0,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary_payload, indent=2))

    if not args.quiet:
        mode = "full" if not skip_permutation else "slim (permutation skipped)"
        print("SpatioLD pipeline complete.")
        print(f"Mode: {mode}")
        print(f"Metadata: {args.metadata}")
        print(f"Expression: {args.expression}")
        print(f"Output dir: {output_dir}")
        print(f"Cells used: {expr_aligned.shape[0]}")
        print(f"Genes after filter: {expr_aligned.shape[1]}")
        print(f"Modeled genes: {expr_model.shape[1]}")
        print(f"Radii: {radii}")


def run_cluster_pipeline(args: argparse.Namespace) -> None:
    if not args.metadata.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}")
    if not args.expression.exists():
        raise FileNotFoundError(f"Expression file not found: {args.expression}")

    radii = _parse_radii(args.radii)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_raw = _load_metadata(
        args.metadata,
        cell_id_col=args.cell_id_col,
        x_col=args.x_col,
        y_col=args.y_col,
        cell_type_col=None,
    )
    expr_raw = _load_expression(args.expression)
    expr = _orient_expression(expr_raw, meta_raw.index)

    expr_filt = preprocess_expression_matrix(
        expr,
        min_fraction_expressed=args.min_fraction_expressed,
        min_genes_per_cell=args.min_genes_per_cell,
    )
    expr_aligned, meta_aligned = align_expression_and_metadata(expr_filt, meta_raw)
    if expr_aligned.empty or meta_aligned.empty:
        raise ValueError("No overlapping cells found between metadata and expression after filtering/alignment.")

    n_top_hvg = int(max(2, min(args.n_top_hvg, expr_aligned.shape[1])))
    hvg_df = _select_hvg_with_fallback(
        expr_aligned,
        n_top_hvg=n_top_hvg,
        hvg_flavor=args.hvg_flavor,
        quiet=args.quiet,
    )
    hvg_genes = hvg_df.index.astype(str).tolist()
    if len(hvg_genes) < 2:
        raise ValueError("Need at least 2 HVGs for leave-one-gene-out clustering.")

    expr_hvg = expr_aligned.loc[:, hvg_genes].copy()
    coords_df = meta_aligned[[args.x_col, args.y_col]].copy()

    summaries: list[dict[str, float | str | int]] = []
    cluster_meta_by_gene_df = pd.DataFrame(index=expr_hvg.index.astype(str))
    n_genes = expr_hvg.shape[1]
    for k, gene in enumerate(expr_hvg.columns, start=1):
        if not args.quiet:
            print(f"Leave-one-gene-out fit {k}/{n_genes}: {gene}")

        expr_wo_gene = expr_aligned.copy()
        expr_wo_gene.drop(columns=[gene], inplace=True)

        if args.cluster_method != "scanpy-leiden":
            raise ValueError(f"Unsupported cluster method: {args.cluster_method}")
        cluster_labels = _cluster_cells_scanpy_leiden(
            expr_wo_gene,
            random_state=args.random_state,
            n_neighbors=args.cluster_n_neighbors,
            n_pcs=args.cluster_n_pcs,
            resolution=args.cluster_resolution,
            n_clusters=args.cluster_n_clusters,
        )
        cluster_meta_by_gene_df[str(gene)] = cluster_labels.reindex(cluster_meta_by_gene_df.index).astype(str)

        ld_df = compute_local_diversity_multi_radius(
            coords_df.loc[cluster_labels.index],
            cluster_labels,
            radii=radii,
        )
        cluster_meta = pd.DataFrame({"cluster_label": cluster_labels})

        shared = prepare_shared_components(
            response_matrix=ld_df.values,
            metadata_df=cluster_meta,
            radius_values=radii,
            cell_type_col="cluster_label",
            radius_mode=args.radius_mode,
            poly_degree=args.poly_degree,
            n_radius_knots=args.n_radius_knots,
            spline_degree=args.spline_degree,
            normalize_by=args.regression_normalize_by,
            normalize_by_global_entropy=not args.no_regression_entropy_normalize,
        )
        fit_res = fit_single_gene_radius_model(
            gene_values=expr_hvg.loc[cluster_labels.index, gene].values,
            shared=shared,
            cluster_robust=not args.no_cluster_robust,
        )

        coef = fit_res["coef"]
        se = fit_res["se"]
        pval = fit_res["pval"]
        beta_gene = float(coef["gene"])
        se_gene = float(se["gene"])
        summaries.append(
            {
                "gene": str(gene),
                "beta_gene": beta_gene,
                "se_gene": se_gene,
                "pval_gene": float(pval["gene"]),
                "t_gene": beta_gene / se_gene if se_gene != 0 else np.nan,
                "r2": float(fit_res["fit"].rsquared),
                "adj_r2": float(fit_res["fit"].rsquared_adj),
                "aic": float(fit_res["fit"].aic),
                "bic": float(fit_res["fit"].bic),
                "n_clusters": int(cluster_labels.nunique()),
            }
        )

    results_df = pd.DataFrame(summaries).sort_values("pval_gene").reset_index(drop=True)
    hvg_out = hvg_df.copy()
    hvg_out["gene"] = hvg_out.index.astype(str)
    hvg_out = hvg_out.reset_index(drop=True)

    results_df.to_csv(output_dir / "cluster_gene_ld_model_results.csv", index=False)
    hvg_out.to_csv(output_dir / "hvg_selected.csv", index=False)
    cluster_meta_by_gene_df.to_csv(output_dir / "cluster_meta_by_gene.csv")
    _write_args_snapshot(args, output_dir, radii, permutation_enabled=False)

    summary_payload = {
        "workflow": "cluster_ld_leave_one_gene_out",
        "n_cells": int(expr_aligned.shape[0]),
        "n_genes_after_filter": int(expr_aligned.shape[1]),
        "n_top_hvg_requested": int(args.n_top_hvg),
        "n_hvg_used": int(expr_hvg.shape[1]),
        "cluster_method": str(args.cluster_method),
        "cluster_n_clusters": (
            int(args.cluster_n_clusters) if args.cluster_n_clusters is not None else None
        ),
        "cluster_resolution": float(args.cluster_resolution),
        "cluster_n_neighbors": int(args.cluster_n_neighbors),
        "cluster_n_pcs": int(args.cluster_n_pcs),
        "n_radii": int(len(radii)),
        "radii": radii,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary_payload, indent=2))

    if not args.quiet:
        print("SpatioLD cluster-label workflow complete.")
        print(f"Metadata: {args.metadata}")
        print(f"Expression: {args.expression}")
        print(f"Output dir: {output_dir}")
        print(f"Cells used: {expr_aligned.shape[0]}")
        print(f"Genes after filter: {expr_aligned.shape[1]}")
        print(f"HVGs modeled: {expr_hvg.shape[1]}")
        print(f"Radii: {radii}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


def main_slim() -> None:
    parser = build_slim_parser()
    args = parser.parse_args()
    run_pipeline(args, skip_permutation=True)


def main_cluster() -> None:
    parser = build_cluster_parser()
    args = parser.parse_args()
    run_cluster_pipeline(args)


if __name__ == "__main__":
    main()
