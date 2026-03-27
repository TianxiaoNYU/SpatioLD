from __future__ import annotations

import numpy as np
import pandas as pd
import pytest # type: ignore

from spatiold import SpatioLD

anndata = pytest.importorskip("anndata")


def test_from_arrays_and_storage_cycle() -> None:
    coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    labels = np.array(["A", "A", "B", "B"], dtype=object)

    obj = SpatioLD.from_arrays(coords, labels, cell_ids=["c1", "c2", "c3", "c4"])

    out = obj.compute_local_diversity(radii=[0.1, 2.0], key="ld_test")
    assert out.shape == (4, 2)

    adata = obj.to_anndata()
    assert "ld_test" in adata.obsm
    assert "ld_test_radii" in adata.uns

    recovered = obj.get_result("ld_test")
    assert recovered.shape == (4, 2)
    assert np.allclose(out.values, recovered.values)

    dist = obj.compute_permutation_distribution(n_perm=6, radii=[2.0], n_jobs=1)
    assert dist.shape == (6, 1, 4)


def test_from_anndata_with_obs_coordinates() -> None:
    adata = anndata.AnnData(X=np.empty((3, 0)))
    adata.obs_names = ["c1", "c2", "c3"]
    adata.obs["cell_type"] = ["A", "B", "A"]
    adata.obs["x"] = [0.0, 1.0, 0.5]
    adata.obs["y"] = [0.0, 0.0, 0.8]

    obj = SpatioLD.from_anndata(
        adata,
        label_key="cell_type",
        coord_keys=("x", "y"),
    )

    pvals = obj.compute_permutation_pvals(
        n_perm=10,
        radii=[1.0],
        n_jobs=1,
        key="pvals_test",
    )

    assert pvals.shape == (3, 1)
    assert "pvals_test" in adata.obsm


def test_object_wrappers_for_downstream_pipeline() -> None:
    rng = np.random.default_rng(7)
    n_cells = 24
    n_genes = 6
    idx = [f"c{i}" for i in range(n_cells)]

    meta = pd.DataFrame(
        {
            "x": rng.uniform(0, 100, size=n_cells),
            "y": rng.uniform(0, 100, size=n_cells),
            "cell_type": rng.choice(["A", "B", "C"], size=n_cells),
        },
        index=idx,
    )
    expr = pd.DataFrame(
        rng.normal(size=(n_cells, n_genes)),
        index=idx,
        columns=[f"g{i}" for i in range(n_genes)],
    )

    adata = anndata.AnnData(
        X=expr.values,
        obs=meta.copy(),
        var=pd.DataFrame(index=expr.columns),
    )
    adata.obs_names = expr.index

    obj = SpatioLD.from_anndata(adata, label_key="cell_type", coord_keys=("x", "y"))

    radii = [20.0, 40.0, 60.0]
    ld = obj.compute_local_diversity(radii=radii, key="ld_full")
    perm_stats = obj.compute_permutation_stats(
        n_perm=8,
        radii=radii,
        n_jobs=1,
        pvals_key="pvals_full",
        perm_mean_key="perm_mean_full",
    )
    pvals = perm_stats["pvals"]
    dist = perm_stats["distribution"]

    assert ld.shape == (n_cells, len(radii))
    assert pvals.shape == (n_cells, len(radii))
    assert perm_stats["perm_mean"].shape == (n_cells, len(radii))
    assert dist.shape == (8, len(radii), n_cells)
    assert obj.compute_global_shannon_entropy() >= 0

    summary_ct = obj.summarize_local_diversity_by_cell_type(local_diversity_key="ld_full")
    summary_null = obj.compute_sample_vs_null_summary(dist, local_diversity_key="ld_full")
    assert {"cell_type", "radius", "mean", "std", "n"}.issubset(summary_ct.columns)
    assert {"radius", "sample_mean", "null_mean"}.issubset(summary_null.columns)

    cluster_df, models = obj.cluster_local_diversity_profiles(local_diversity_key="ld_full", k_values=(2, 3))
    mask = obj.build_significance_mask(pvals_key="pvals_full", alpha=0.05)
    shared = obj.prepare_shared_components(local_diversity_key="ld_full", radius_mode="poly", poly_degree=2)
    svg = obj.compute_svg_morans_i(expr, k=5)

    assert set(cluster_df.columns) == {"ld_kmeans_k2", "ld_kmeans_k3"}
    assert set(models.keys()) == {2, 3}
    assert mask.shape == pvals.shape
    assert shared["n_cells"] == n_cells
    assert shared["n_radii"] == len(radii)
    entropy = obj.compute_global_shannon_entropy()
    if entropy > 0:
        assert np.allclose(shared["Y"], ld.values / entropy)
        assert np.isclose(shared["response_normalization_factor"], entropy)
    assert list(svg.columns) == ["gene", "moran_I"]
    assert svg.shape[0] == n_genes
