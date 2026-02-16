# Figures and Reports

## Figure generation

NBT report workflows generate figures from run artifacts and metrics, including:

- embedding similarity heatmaps,
- PCA/UMAP projection plots,
- sentiment comparison charts,
- attractor density comparisons.

Geopolitical reporting uses matplotlib-only plotting for portability.

## Report outputs

Reports are written to timestamped folders under `reports/`.

Typical outputs:

- `report.md` or `report.html`
- `figures/*.png`
- comparison tables (`comparison.csv`, `comparison.md` for multi-run analysis)

## Citation in dissertation writing

When citing results in chapters or appendices, reference:

1. run folder path,
2. artifact path,
3. metric entry or derived figure.

This provides transparent provenance from claim to computational trace.
