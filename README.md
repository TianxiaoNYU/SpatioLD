# SpatioLD

`SpatioLD` is a standalone Python package for spatial local diversity analysis in spatial transcriptomics and related spatial omics data.

This package includes:

- Fundamental local-diversity computation and permutation inference
- A unified `SpatioLD` object workflow for downstream analysis
- Updated gene-radius modeling pipeline (spline/poly basis + per-gene fitting)
- Pipeline-level utilities for clustering, summaries, SVG scoring
- Visualization helpers for key outputs

## Installation

From the `SpatioLD/` folder:

```bash
pip install -e .
```

### Recreate Conda Environment Snapshot

A full export from the `spatiometrics` conda environment is saved at:

- `requirements-spatiold.txt`

This snapshot was exported on `osx-arm64`.

Recreate a matching environment:

```bash
conda create --name spatiold --file requirements-spatiold.txt
conda activate spatiold
```

Install this package in editable mode inside that environment:

```bash
pip install -e .
```

For the exact edge-list local-diversity backend, install the optional fast
dependency if your environment does not already include Numba:

```bash
pip install -e ".[fast]"
```

## Quick Start

### 1) Functional API

```python
import pandas as pd
import spatiold as sld

coords = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 0, 1]})
labels = pd.Series(["A", "A", "B", "B"], index=["c1", "c2", "c3", "c4"])

ld_df = sld.compute_local_diversity_multi_radius(coords, labels, radii=[0.1, 2.0])
perm_stats = sld.compute_nd_permutation_stats(
    coords,
    labels,
    n_perm=10,
    radii=[2.0],
    pval_pooling="neighborhood_size",
)
pvals_mixing_df = perm_stats["pvals_mixing"]
pvals_segregation_df = perm_stats["pvals_segregation"]
pvals_two_sided_df = perm_stats["pvals_two_sided"]
perm_dist = perm_stats["distribution"]

# distance-weighted local diversity
ld_weighted = sld.compute_local_diversity_multi_radius(
    coords,
    labels,
    radii=[2.0],
    include_self=False,
    kernel="gaussian",
)
# For exact full-graph Gaussian weights, add `kernel_support=np.inf`.

# force the exact edge-list aggregation backend when finite support is used
ld_fast = sld.compute_local_diversity_multi_radius(
    coords,
    labels,
    radii=[2.0],
    kernel="gaussian",
    kernel_support=2.0,
    aggregation_backend="edge",
)
```

### 2) SpatioLD Object Workflow

```python
from spatiold import SpatioLD

coords = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = ["A", "A", "B", "B"]

obj = SpatioLD.from_arrays(coords=coords, labels=labels, cell_ids=["c1", "c2", "c3", "c4"])
ld_df = obj.compute_local_diversity(radii=[0.1, 2.0], store=True)

# weighted graph version with w_ii = 0
ld_weighted = obj.compute_local_diversity(
    radii=[2.0],
    include_self=False,
    kernel="gaussian",
    store=False,
)
# Default Gaussian support stays local for speed; use `kernel_support=np.inf`
# if you need the exact full graph.
```

### 3) Object-Centric SlideTag-Style Pipeline

```python
import spatiold as sld

# assume obj = SpatioLD.from_arrays(...) or SpatioLD.from_anndata(...)
ld_df = obj.compute_local_diversity(radii=[30, 60, 90], key="ld_full")
perm_stats = obj.compute_permutation_stats(
    n_perm=10,
    radii=[30, 60, 90],
    pval_pooling="neighborhood_size",
    perm_mean_key="ld_perm_mean",
)
pvals_mixing_df = perm_stats["pvals_mixing"]
pvals_segregation_df = perm_stats["pvals_segregation"]
pvals_two_sided_df = perm_stats["pvals_two_sided"]
zscore_df = perm_stats["zscore"]
perm_dist = perm_stats["distribution"]

# summaries from object-held results
summary_ct = obj.summarize_local_diversity_by_cell_type(local_diversity_key="ld_full")
summary_null = obj.compute_sample_vs_null_summary(perm_dist, local_diversity_key="ld_full")

# per-gene radius model using matched-null z scores
shared = obj.prepare_shared_components(
    local_diversity_key="spatiold_local_diversity_zscore",
    radius_mode="poly",
    poly_degree=3,
)
results_df, fit_objects = sld.fit_all_genes(expr_df, shared)
# For large slides, this switches to the shared batched all-gene solver:
results_df_only, _ = sld.fit_all_genes(expr_df, shared, store_fit_objects=False)
# You can also force the batched solver explicitly:
results_df_batch, _ = sld.fit_all_genes(expr_df, shared, store_fit_objects=False, method="batch")

# true joint model: all genes enter the same OLS, still adjusting for cell type and radius
joint_fit = sld.fit_joint_gene_radius_model(expr_df, shared)
joint_gene_df = joint_fit["gene_summary"]

# slide-level cell-type model (fit once per slide)
ct_fit = sld.fit_slide_level_cell_type_radius_model(shared)
ct_effects = sld.summarize_slide_level_cell_type_effects(ct_fit, shared)

# plotting
ax = sld.plot_gene_effect_volcano(results_df)
svg_df = obj.compute_svg_morans_i(expr_df, k=15)
```

## Terminal Pipeline Runner

After installation, you can run the full pipeline directly from terminal:

```bash
spatiold \
  --metadata /path/to/metadata.csv \
  --expression /path/to/expression.csv \
  --output-dir /path/to/output \
  --radii 30 60 90 120 150 180 210 240 270 \
  --n-perm 10 \
  --pval-pooling neighborhood-size \
  --min-genes-per-cell 10
```

To switch from the default single-gene regression workflow to the joint all-gene OLS in the CLI, add:

```bash
  --gene-model-mode joint
```

Or run directly from a single AnnData file:

```bash
spatiold \
  --input-h5ad /path/to/data.h5ad \
  --output-dir /path/to/output \
  --cell-type-col cell_type \
  --x-col x --y-col y
```

To run the full pipeline while skipping only gene-radius regression, use:

```bash
spatiold-lite \
  --metadata /path/to/metadata.csv \
  --expression /path/to/expression.csv \
  --output-dir /path/to/output \
  --radii 30 60 90 120 150 180 210 240 270 \
  --n-perm 10 \
  --pval-pooling neighborhood-size \
  --min-genes-per-cell 10
```

`spatiold-lite` keeps preprocessing, local diversity, permutation inference, and slide-level modeling, but skips both gene-radius regression and SVG/Moran's I scoring. All CLI workflows also write an aligned `metadata.csv` in the output directory for downstream plotting.

Permutation p-values now default to pooling null draws across cells with the
same neighborhood size at each radius. This keeps the CSR null aligned with
the local neighborhood size while allowing much smaller `n_perm` values than
the legacy per-cell-only comparison.

For metadata without cell-type annotations, run the cluster-label workflow:

```bash
spatiold-cluster \
  --metadata /path/to/metadata_xy_only.csv \
  --expression /path/to/expression.csv \
  --output-dir /path/to/output \
  --radii 30 60 90 120 \
  --n-top-hvg 100 \
  --cluster-n-clusters 8
```

`spatiold-cluster` uses top HVGs, then for each gene performs leave-one-gene-out clustering (default `scanpy-leiden`) to generate cluster labels, computes LD from those labels, and fits the single-gene LD association model. Main outputs:

- `cluster_gene_ld_model_results.csv`
- `hvg_selected.csv`
- `cluster_meta_by_gene.csv` (rows: cells, columns: gene IDs, values: leave-one-gene-out cluster labels)
- `metadata.csv`
- `run_config.json`
- `run_summary.json`

For a faster variant, add `--simplify`:

```bash
spatiold-cluster \
  --metadata /path/to/metadata_xy_only.csv \
  --expression /path/to/expression.csv \
  --output-dir /path/to/output \
  --radii 30 60 90 120 \
  --n-top-hvg 100 \
  --cluster-n-clusters 8 \
  --simplify
```

In simplify mode, clustering runs once on the full filtered expression matrix, and the resulting labels are reused for all per-gene downstream modeling steps. Output file names remain the same.

Clustering controls:

- `--cluster-n-clusters` directly sets cluster count and takes priority over `--cluster-resolution`.
- `--cluster-resolution` is used when `--cluster-n-clusters` is not provided.

Equivalent module form:

```bash
python -m spatiold.cli \
  --metadata /path/to/metadata.csv \
  --expression /path/to/expression.csv \
  --output-dir /path/to/output \
  --radii 30 60 90 120 150 180 210 240 270
```

Required inputs:

- either `--input-h5ad` or both `--metadata` + `--expression`
- `--output-dir`: directory for outputs
- `--radii`: radius grid

Common optional inputs (dataset-dependent):

- `--cell-id-col` if cell IDs are not in index/`unique_id`
- `--x-col`, `--y-col`, `--cell-type-col` if metadata columns use different names
- `--spatial-kernel gaussian` and optional `--kernel-support` for distance-weighted local diversity
  (default `--kernel-support 1` keeps the same fixed neighborhood as the legacy radius mode;
  `--kernel-support inf` gives the exact full Gaussian graph)
- `--ld-backend auto|csr|edge` to choose local-diversity aggregation; `auto`
  uses the exact edge-list backend when Numba is available, support is finite,
  and the cell-by-label accumulator is reasonably sized
- `--n-jobs` defaults to 1 for permutation inference; this is recommended for
  the edge backend because geometry is reused in-process
- `--perm-block-size` defaults to 32 and controls how many global label
  permutations are evaluated per compiled edge-list pass when `--n-jobs 1`
- `--exclude-self` to match the weighted-graph convention `w_ii = 0`
- `--spatial-key` to read coordinates from `adata.obsm[spatial_key]` in `.h5ad`
- `--h5ad-layer` to use an AnnData layer instead of `adata.X`
- `--cell-size-col` to include cell size (or another numeric metadata column) as a regression covariate
- `--n-perm`, `--n-model-genes` for runtime control
- full pipeline regression now uses matched-null z scores by default; the legacy
  global-entropy normalization is no longer the default

CLI now also writes slide-level cell-type model outputs:

- `slide_cell_type_radius_model_terms.csv`
- `slide_cell_type_effects.csv`
- `local_diversity_pvals_mixing.csv`
- `local_diversity_pvals_segregation.csv`
- `local_diversity_pvals_two_sided.csv`
- `local_diversity_perm_std.csv`

## Main Modules

- `src/spatiold/diversity.py`: local-diversity computation
- `src/spatiold/permutation.py`: p-values, null mean, full permutation distribution
- `src/spatiold/core.py`: `SpatioLD` object API
- `src/spatiold/modeling.py`: regression helpers from prior workflow
- `src/spatiold/pipeline.py`: updated SlideTag-style fundamental pipeline functions
- `src/spatiold/plotting.py`: visualization utilities for pipeline outputs

## Demos

- `demo/synthetic_quickstart.py`
- `demo/anndata_quickstart.py` (compatibility example)
- `demo/slidetag_style_pipeline.py` (uses `example_data/SlideTag_HumanCortex.csv` metadata and synthetic expression, default 1000 genes)

Run the SlideTag-style demo:

```bash
python demo/slidetag_style_pipeline.py
# optional plotting
python demo/slidetag_style_pipeline.py --plot
```

## Tests

From `SpatioLD/`:

```bash
pytest
```

## Notes on Compatibility

- Explicit local mixing, segregation, and two-sided permutation p-values are available through `compute_nd_permutation_stats`.
- `compute_nd_permutation_mean` remains available.
- `compute_neighborhood_diversity` remains available as an alias.
- New `compute_nd_permutation_distribution` supports downstream CI/null-curve plotting in the updated pipeline.
- `SpatioLD` is compatible with AnnData through `SpatioLD.from_anndata(...)`.
