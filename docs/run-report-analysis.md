# Run Report & Analysis

Use the report command to summarize a run folder with analysis utilities and artifact citations.

```bash
nbt report runs/<timestamp>_<experiment>
```

Output:

- `report.md` in the run directory
- analysis artifacts in `artifacts/analysis/`

Included analysis utilities:

- embedding topology projections (PCA + UMAP/fallback)
- attention entropy metrics
- KL divergence helpers for token distributions
- coherence proxies (self-consistency, repetition, temporal reference stability)
- image diversity proxies (LPIPS optional; perceptual hash fallback)
