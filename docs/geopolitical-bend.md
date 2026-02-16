# Geopolitical Bend

The Geopolitical Bend experiment is available as `geopolitical-bend` and now implements all three dissertation-inspired phases end to end.

## CLI

```bash
nbt geopolitical describe
nbt geopolitical run --config configs/geopolitical.example.yaml
nbt geopolitical report <run_dir>
nbt geopolitical compare <run1> <run2>
```

## Config schema includes

- model identifiers
- governance concepts
- contradictory prompt pairs
- justice attractor token sets
- logging options and random seed

## Implemented phases

1. **Ontological Mapping**
   - Extracts governance-token embeddings (LLM + diffusion-CLIP fallback-aware paths)
   - Computes cosine similarity matrices
   - Saves embeddings (`.npy`), similarity tables (`.csv`), and PCA/UMAP outputs (`.npy` + plot images when matplotlib is available)
2. **Governance Dissonance Stress Test**
   - Runs contradictory prompt pairs
   - Computes refusal detection, sentiment polarity heuristic, structural causality heuristic, and KL divergence proxies
   - Saves structured JSON and CSV summaries
3. **Justice Attractor Injection**
   - Compares baseline vs attractor-injected generations
   - Computes attractor token density changes
   - Saves side-by-side comparison text files and summary JSON

Artifacts are stored under `artifacts/geopolitical/phase_*` with deterministic seeding logged for reproducibility.

Example config: `configs/geopolitical.example.yaml`.


Reports and comparisons are written under a timestamped `reports/` directory.
