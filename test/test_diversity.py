from __future__ import annotations

import numpy as np
import pandas as pd

from spatiold import compute_local_diversity, compute_local_diversity_multi_radius


def test_single_radius_entropy_limits() -> None:
    coords = pd.DataFrame({"x": [0, 0, 1, 1], "y": [0, 1, 0, 1]})
    labels = pd.Series(["A", "A", "B", "B"], index=["c1", "c2", "c3", "c4"])

    # Only itself in neighborhood -> entropy 0
    tiny = compute_local_diversity(coords, labels, radius=0.01)
    assert np.allclose(tiny, 0.0)

    # All cells in neighborhood -> two classes with p=0.5 each -> entropy 1 bit
    big = compute_local_diversity(coords, labels, radius=10.0)
    assert np.allclose(big, 1.0)


def test_multi_radius_returns_dataframe_with_cell_ids() -> None:
    coords = pd.DataFrame({"x": [0, 0, 1], "y": [0, 1, 0]}, index=["a", "b", "c"])
    labels = pd.Series(["A", "A", "B"], index=coords.index)

    out = compute_local_diversity_multi_radius(coords, labels, radii=[0.1, 2.0])

    assert out.shape == (3, 2)
    assert list(out.index) == ["a", "b", "c"]
    assert list(out.columns) == [0.1, 2.0]
    assert (out.values >= 0).all()


def test_gaussian_kernel_matches_manual_weighted_entropy() -> None:
    coords = pd.DataFrame({"x": [0.0, 0.5, 3.0], "y": [0.0, 0.0, 0.0]})
    labels = pd.Series(["A", "A", "B"])

    out = compute_local_diversity(
        coords,
        labels,
        radius=1.0,
        include_self=False,
        kernel="gaussian",
        kernel_support=np.inf,
    )

    w_same = np.exp(-0.5 * (0.5 / 1.0) ** 2)
    w_far_from_first = np.exp(-0.5 * (3.0 / 1.0) ** 2)
    p_a_first = w_same / (w_same + w_far_from_first)
    p_b_first = w_far_from_first / (w_same + w_far_from_first)
    expected_first = -(p_a_first * np.log2(p_a_first) + p_b_first * np.log2(p_b_first))

    w_far_from_second = np.exp(-0.5 * (2.5 / 1.0) ** 2)
    p_a_second = w_same / (w_same + w_far_from_second)
    p_b_second = w_far_from_second / (w_same + w_far_from_second)
    expected_second = -(p_a_second * np.log2(p_a_second) + p_b_second * np.log2(p_b_second))

    assert np.isclose(out[0], expected_first)
    assert np.isclose(out[1], expected_second)
    assert np.isclose(out[2], 0.0)


def test_gaussian_default_support_stays_local_and_fast() -> None:
    coords = pd.DataFrame({"x": [0.0, 0.5, 3.0], "y": [0.0, 0.0, 0.0]})
    labels = pd.Series(["A", "A", "B"])

    out = compute_local_diversity(
        coords,
        labels,
        radius=1.0,
        include_self=False,
        kernel="gaussian",
    )

    assert np.allclose(out, 0.0)


def test_indicator_kernel_is_backward_compatible() -> None:
    coords = pd.DataFrame({"x": [0.0, 0.0, 1.0], "y": [0.0, 1.0, 0.0]})
    labels = pd.Series(["A", "A", "B"])

    legacy = compute_local_diversity_multi_radius(coords, labels, radii=[0.5, 2.0])
    explicit = compute_local_diversity_multi_radius(
        coords,
        labels,
        radii=[0.5, 2.0],
        kernel="indicator",
    )

    assert legacy.equals(explicit)


def test_edge_backend_matches_csr_indicator() -> None:
    coords = pd.DataFrame(
        {"x": [0.0, 0.3, 1.0, 1.8, 2.7], "y": [0.0, 0.4, 0.9, 0.1, 1.1]},
        index=[f"c{i}" for i in range(5)],
    )
    labels = pd.Series(["A", "B", "A", "C", "B"], index=coords.index)

    csr = compute_local_diversity_multi_radius(
        coords,
        labels,
        radii=[0.5, 1.5, 3.0],
        aggregation_backend="csr",
    )
    edge = compute_local_diversity_multi_radius(
        coords,
        labels,
        radii=[0.5, 1.5, 3.0],
        aggregation_backend="edge",
    )

    assert edge.index.equals(csr.index)
    assert list(edge.columns) == list(csr.columns)
    assert np.allclose(edge.values, csr.values)


def test_edge_backend_matches_csr_gaussian() -> None:
    coords = pd.DataFrame({"x": [0.0, 0.5, 1.5, 3.0], "y": [0.0, 0.0, 0.0, 0.0]})
    labels = pd.Series(["A", "A", "B", "B"])

    csr = compute_local_diversity_multi_radius(
        coords,
        labels,
        radii=[1.0, 2.0],
        include_self=False,
        kernel="gaussian",
        kernel_support=2.0,
        aggregation_backend="csr",
    )
    edge = compute_local_diversity_multi_radius(
        coords,
        labels,
        radii=[1.0, 2.0],
        include_self=False,
        kernel="gaussian",
        kernel_support=2.0,
        aggregation_backend="edge",
    )

    assert np.allclose(edge.values, csr.values)
