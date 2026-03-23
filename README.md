# SpatioLD

`SpatioLD` is a standalone Python package for spatial local diversity analysis in spatial transcriptomics and related spatial omics data.

This package now reflects the newer `src/slidetag.ipynb` pipeline and includes:

- Fundamental local-diversity computation and permutation inference
- AnnData-compatible object API
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

### 2) AnnData object API

```python
import numpy as np
import anndata as ad
from spatiold import SpatioLD

adata = ad.AnnData(X=np.empty((4, 0)))
adata.obs_names = ["c1", "c2", "c3", "c4"]
adata.obs["cell_type"] = ["A", "A", "B", "B"]
adata.obsm["spatial"] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

obj = SpatioLD.from_anndata(adata, label_key="cell_type")
ld_df = obj.compute_local_diversity(radii=[0.1, 2.0], store=True)
```

### 3) Object-Centric SlideTag-Style Pipeline

```python
import spatiold as sld

# assume obj = SpatioLD.from_anndata(...)
ld_df = obj.compute_local_diversity(radii=[30, 60, 90], key="ld_full")
pvals_df = obj.compute_permutation_pvals(n_perm=100, radii=[30, 60, 90], key="ld_pvals")
perm_dist = obj.compute_permutation_distribution(n_perm=100, radii=[30, 60, 90])

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
)
results_df, fit_objects = sld.fit_all_genes(expr_df, shared)

# plotting
ax = sld.plot_gene_effect_volcano(results_df)
svg_df = obj.compute_svg_morans_i(expr_df, k=15)
```

## Main Modules

- `src/spatiold/diversity.py`: local-diversity computation
- `src/spatiold/permutation.py`: p-values, null mean, full permutation distribution
- `src/spatiold/core.py`: `SpatioLD` object (AnnData-first)
- `src/spatiold/modeling.py`: regression helpers from prior workflow
- `src/spatiold/pipeline.py`: updated SlideTag-style fundamental pipeline functions
- `src/spatiold/plotting.py`: visualization utilities for pipeline outputs

## Demos

- `demo/synthetic_quickstart.py`
- `demo/anndata_quickstart.py`
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
