# SpatioLD API Reference

## Core Object

### `spatiold.SpatioLD`

AnnData-first object.

Constructors:

- `SpatioLD.from_anndata(...)`
- `SpatioLD.from_arrays(...)`

Main methods:

- `compute_local_diversity(...)`
- `compute_permutation_stats(...)` (single-pass p-values + null mean + distribution)
- `compute_permutation_pvals(...)`
- `compute_permutation_mean(...)`
- `compute_permutation_distribution(...)`
- `compute_global_shannon_entropy(...)`
- `summarize_local_diversity_by_cell_type(...)`
- `compute_sample_vs_null_summary(...)`
- `cluster_local_diversity_profiles(...)`
- `build_significance_mask(...)`
- `prepare_shared_components(...)` (regression response is entropy-normalized by default)
- `fit_slide_level_cell_type_radius_model(...)`
- `summarize_slide_level_cell_type_effects(...)`
- `summarize_model_terms(...)`
- `compute_svg_morans_i(...)`
- `get_coords_df(...)`
- `get_result(key)`
- `to_anndata()`

## Local Diversity / Permutation

- `compute_local_diversity(coords, labels, radius, ...)`
- `compute_local_diversity_multi_radius(coords, labels, radii, ...)`
- `compute_nd_permutation_pvals(xy, labels, n_perm, ...)`
- `compute_nd_permutation_mean(xy, labels, n_perm, ...)`
- `compute_nd_permutation_distribution(xy, labels, n_perm, ...)`
- `compute_nd_permutation_stats(xy, labels, n_perm, ...)` (single-pass combined output)
- `DEFAULT_RADII`

## Updated SlideTag-Style Pipeline

Fundamental utilities:

- `compute_global_shannon_entropy(labels)`
- `preprocess_expression_matrix(expr_df, ...)`
- `align_expression_and_metadata(expr_df, metadata_df)`
- `summarize_local_diversity_by_cell_type(local_diversity_df, metadata_df, ...)`
- `compute_sample_vs_null_summary(local_diversity_df, permutation_distribution, ...)`
- `cluster_local_diversity_profiles(local_diversity_df, ...)`
- `build_significance_mask(pvals_df, alpha=0.05)`

Gene-radius model utilities:

- `make_spline_basis(radius_values, ...)`
- `prepare_shared_components(response_matrix, metadata_df, radius_values, ..., covariate_cols=None, normalize_by=None, normalize_by_global_entropy=True)`
- `fit_single_gene_radius_model(gene_values, shared, ...)`
- `reconstruct_radius_effect(fit_result, shared, ...)`
- `fit_all_genes(expr_df, shared, ...)`
- `fit_slide_level_cell_type_radius_model(shared, ...)`
- `summarize_slide_level_cell_type_effects(fit_result, shared, ...)`
- `summarize_model_terms(fit_result)`

Spatial/HVG helpers:

- `compute_svg_morans_i(expr_df, coords_df, ...)`
- `compute_hvg_scanpy(expr_df, ...)` (requires Scanpy)

## Visualization

- `plot_spatial_cell_types(...)`
- `plot_mean_diversity_by_cell_type(...)`
- `plot_sample_vs_null_curve(...)`
- `plot_kmeans_spatial_maps(...)`
- `plot_significant_diversity_maps(...)`
- `plot_gene_effect_volcano(...)`
- `plot_gene_set_venn(...)` (requires `matplotlib-venn`)

## Previous Regression Helpers (Retained)

- `prepare_radius_design(...)`
- `fit_lasso_for_celltype(...)`
- `fit_ols_for_celltype(...)`
- `fit_lasso_for_radius(...)`
- `fit_ols_for_radius(...)`
- `top_abs_terms(...)`
