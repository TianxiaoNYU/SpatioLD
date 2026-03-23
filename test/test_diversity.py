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
