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

## Quick Start

### 1) Functional API

```python
import pandas as pd
import spatiold as sld

coords = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 0, 1]})
labels = pd.Series(["A", "A", "B", "B"], index=["c1", "c2", "c3", "c4"])

ld_df = sld.compute_local_diversity_multi_radius(coords, labels, radii=[0.1, 2.0])
pval_df = sld.compute_nd_permutation_pvals(coords, labels, n_perm=200, radii=[2.0])
perm_dist = sld.compute_nd_permutation_distribution(coords, labels, n_perm=200, radii=[2.0])
```

### 2) SpatioLD Object Workflow

```python
from spatiold import SpatioLD

coords = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = ["A", "A", "B", "B"]

obj = SpatioLD.from_arrays(coords=coords, labels=labels, cell_ids=["c1", "c2", "c3", "c4"])
ld_df = obj.compute_local_diversity(radii=[0.1, 2.0], store=True)
```

### 3) Object-Centric SlideTag-Style Pipeline

```python
import spatiold as sld

# assume obj = SpatioLD.from_arrays(...) or SpatioLD.from_anndata(...)
ld_df = obj.compute_local_diversity(radii=[30, 60, 90], key="ld_full")
perm_stats = obj.compute_permutation_stats(
    n_perm=100,
    radii=[30, 60, 90],
    pvals_key="ld_pvals",
    perm_mean_key="ld_perm_mean",
)
pvals_df = perm_stats["pvals"]
perm_dist = perm_stats["distribution"]

# summaries and clustering from object-held results
entropy = obj.compute_global_shannon_entropy()
summary_ct = obj.summarize_local_diversity_by_cell_type(local_diversity_key="ld_full")
summary_null = obj.compute_sample_vs_null_summary(perm_dist, local_diversity_key="ld_full")
labels_df, models = obj.cluster_local_diversity_profiles(local_diversity_key="ld_full", k_values=(2, 3, 4))
sig_mask = obj.build_significance_mask(pvals_key="ld_pvals", alpha=0.05)

# per-gene radius model
shared = obj.prepare_shared_components(
    local_diversity_key="ld_full",
    radius_mode="poly",
    poly_degree=3,
    # default: divide local diversity by global Shannon entropy
    normalize_by_global_entropy=True,
)
results_df, fit_objects = sld.fit_all_genes(expr_df, shared)

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
spatiold-pipeline \
  --metadata /path/to/metadata.csv \
  --expression /path/to/expression.csv \
  --output-dir /path/to/output \
  --radii 30 60 90 120 150 180 210 240 270 --n-perm 10 --min-genes-per-cell 10
```

Or run directly from a single AnnData file:

```bash
spatiold-pipeline \
  --input-h5ad /path/to/data.h5ad \
  --output-dir /path/to/output \
  --cell-type-col cell_type \
  --x-col x --y-col y
```

For faster iteration without permutation inference, use slim mode:

```bash
spatiold-pipeline-slim \
  --metadata /path/to/metadata.csv \
  --expression /path/to/expression.csv \
  --output-dir /path/to/output \
  --radii 30 60 90 120 150 180 210 240 270 --min-genes-per-cell 10
```

`spatiold-pipeline-slim` skips permutation p-value/null computation, while keeping preprocessing, local diversity, clustering, slide-level modeling, gene-radius modeling, and SVG scoring. Permutation-dependent output files are still written with placeholder values.

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
- `run_config.json`
- `run_summary.json`

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
- `--spatial-key` to read coordinates from `adata.obsm[spatial_key]` in `.h5ad`
- `--h5ad-layer` to use an AnnData layer instead of `adata.X`
- `--cell-size-col` to include cell size (or another numeric metadata column) as a regression covariate
- `--n-perm`, `--n-model-genes` for runtime control
- `--regression-normalize-by` or `--no-regression-entropy-normalize` to control
  response normalization in gene-radius regression

CLI now also writes slide-level cell-type model outputs:

- `slide_cell_type_radius_model_terms.csv`
- `slide_cell_type_effects.csv`

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

- `compute_nd_permutation_pvals` and `compute_nd_permutation_mean` are preserved.
- `compute_neighborhood_diversity` remains available as an alias.
- New `compute_nd_permutation_distribution` supports downstream CI/null-curve plotting in the updated pipeline.
- `SpatioLD` is compatible with AnnData through `SpatioLD.from_anndata(...)`.
