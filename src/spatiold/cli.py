"""Command-line pipeline runner for SpatioLD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .core import SpatioLD
from .pipeline import align_expression_and_metadata, fit_all_genes, preprocess_expression_matrix

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
    cell_type_col: str,
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

    required = [x_col, y_col, cell_type_col]
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


def _write_args_snapshot(args: argparse.Namespace, output_dir: Path, radii: list[float]) -> None:
    payload = vars(args).copy()
    payload["radii"] = radii
    payload["metadata"] = str(args.metadata)
    payload["expression"] = str(args.expression)
    payload["output_dir"] = str(args.output_dir)
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SpatioLD pipeline on one sample from metadata + expression tables."
    )
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

    parser.add_argument("--n-perm", type=int, default=100, help="Permutation count for p-values and null summaries.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance alpha for p-value mask.")

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

    parser.add_argument("--save-permutation-distribution", action="store_true", help="Also save full permutation distribution to `.npz`.")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress messages.")
    return parser


def run_pipeline(args: argparse.Namespace) -> None:
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

    summary_ct = obj.summarize_local_diversity_by_cell_type(local_diversity_key=ld_key)
    summary_null = obj.compute_sample_vs_null_summary(perm_dist, local_diversity_key=ld_key)
    cluster_labels_df, _ = obj.cluster_local_diversity_profiles(local_diversity_key=ld_key, k_values=args.k_values)
    sig_mask_df = obj.build_significance_mask(pvals_key=pval_key, alpha=args.alpha)

    n_model_genes = max(1, min(int(args.n_model_genes), expr_aligned.shape[1]))
    var_rank = expr_aligned.var(axis=0).sort_values(ascending=False)
    model_genes = var_rank.head(n_model_genes).index
    expr_model = expr_aligned.loc[:, model_genes].copy()

    shared = obj.prepare_shared_components(
        local_diversity_key=ld_key,
        radius_mode=args.radius_mode,
        poly_degree=args.poly_degree,
        n_radius_knots=args.n_radius_knots,
        spline_degree=args.spline_degree,
        normalize_by=args.regression_normalize_by,
        normalize_by_global_entropy=not args.no_regression_entropy_normalize,
    )
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
    results_df.to_csv(output_dir / "gene_radius_model_results.csv", index=False)
    svg_df.to_csv(output_dir / "svg_morans_i.csv", index=False)

    if args.save_permutation_distribution:
        np.savez_compressed(output_dir / "permutation_distribution.npz", permutation_distribution=perm_dist)

    _write_args_snapshot(args, output_dir, radii)

    summary_payload = {
        "n_cells": int(expr_aligned.shape[0]),
        "n_genes_after_filter": int(expr_aligned.shape[1]),
        "n_model_genes": int(expr_model.shape[1]),
        "n_radii": int(len(radii)),
        "radii": radii,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary_payload, indent=2))

    if not args.quiet:
        print("SpatioLD pipeline complete.")
        print(f"Metadata: {args.metadata}")
        print(f"Expression: {args.expression}")
        print(f"Output dir: {output_dir}")
        print(f"Cells used: {expr_aligned.shape[0]}")
        print(f"Genes after filter: {expr_aligned.shape[1]}")
        print(f"Modeled genes: {expr_model.shape[1]}")
        print(f"Radii: {radii}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
