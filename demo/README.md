# Demo scripts

Run from `SpatioLD/` after installation (`pip install -e .`).

```bash
python demo/synthetic_quickstart.py
python demo/anndata_quickstart.py
python demo/slidetag_style_pipeline.py
# with plotting enabled
python demo/slidetag_style_pipeline.py --plot
```

Scripts:

- `synthetic_quickstart.py`: array/DataFrame API + permutation tests.
- `anndata_quickstart.py`: AnnData object workflow with result storage and reload.
- `slidetag_style_pipeline.py`: updated end-to-end pipeline using `example_data/SlideTag_HumanCortex.csv` metadata and synthetic expression (default 1000 genes; model uses 250 by default for speed).
