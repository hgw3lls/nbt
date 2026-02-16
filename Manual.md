# Neural Bending Toolkit Manual

This manual is a practical guide to the Neural Bending Toolkit (NBT), with a complete reference to the user-facing toolkit functions and end-to-end tutorials.

---

## 1) What this toolkit does

NBT helps you:

- run bend-oriented model experiments,
- log events/metrics/artifacts into structured run directories,
- compute standardized post-run analysis,
- classify bend behavior,
- generate reports/figures/theory memos,
- run geopolitical bend workflows,
- prototype modular realtime patches in the console subsystem.

---

## 2) Installation and setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Optional extras by use case:

```bash
# Hugging Face causal LLM workflows
pip install -e .[llm]

# Diffusers workflows
pip install -e .[diffusion]

# StyleGAN workflows
pip install -e .[gan]

# Audio workflows
pip install -e .[audio]
```

---

## 3) Run directory anatomy

Most commands operate on a run directory:

```text
runs/<timestamp>_<experiment>/
  config.yaml
  events.log
  events.jsonl
  metrics.jsonl
  artifacts/
  analysis/
  reports/
```

You usually create this directory with `nbt run ...`.

---

## 4) CLI function reference (complete)

## Core commands

### `nbt list`
**Purpose:** List discovered experiments.

### `nbt describe <experiment>`
**Purpose:** Show experiment schema/description.

### `nbt run <experiment> --config <path.yaml>`
**Purpose:** Execute one experiment from YAML config.

### `nbt report <run_dir> [--output report.md]`
**Purpose:** Generate a markdown report for a run.

### `nbt curate <run_dir>`
**Purpose:** Build a curated bundle containing key outputs (memo, metrics, classification, reports, top figures).

## Analysis subgroup

### `nbt analyze derive <run_dir>`
**Purpose:** Write derived standardized metrics JSON.

### `nbt analyze classify <run_dir>`
**Purpose:** Write bend classification JSON.

### `nbt analyze all <run_dir>`
**Purpose:** Run derive + classify pipeline.

## Memo subgroup

### `nbt memo build <run_dir>`
**Purpose:** Generate a single theory memo markdown file.

### `nbt memo build-all <runs_root>`
**Purpose:** Generate theory memos for all discovered runs under a root.

## Geopolitical subgroup

### `nbt geopolitical describe`
**Purpose:** Describe the geopolitical experiment schema.

### `nbt geopolitical run --config <path.yaml>`
**Purpose:** Execute the geopolitical bend experiment.

### `nbt geopolitical report <run_dir> [--format markdown|html]`
**Purpose:** Generate geopolitical report in markdown or HTML.

### `nbt geopolitical compare <run_dir_1> <run_dir_2> ...`
**Purpose:** Compare multiple geopolitical runs; emits tabular + summary outputs.

## Console subgroup

### `nbt console serve [--host 127.0.0.1 --port 8000]`
**Purpose:** Start websocket/API server backing the modular console UI.

### `nbt console validate --patch <patch.json> [--patch <patch2.json> ...]`
**Purpose:** Validate one or more console patch graphs.

### `nbt console init [output_path]`
**Purpose:** Generate a starter patch JSON.

### `nbt console patches`
**Purpose:** List starter patches in `patches/`.

---

## 5) Python API function reference (complete public exports)

The sections below cover the exported toolkit functions and classes intended for direct Python usage.

## A) Analysis API (`neural_bending_toolkit.analysis`)

### `attention_entropy(attn, axis=-1, eps=1e-12) -> np.ndarray`
Compute entropy over attention probabilities.

### `js_divergence(p, q, eps=1e-12) -> float`
Compute Jensen-Shannon divergence between two distributions.

### `kl_divergence(p, q, eps=1e-12) -> float`
Compute KL divergence `KL(P || Q)`.

### `self_consistency_score(texts) -> float`
Proxy coherence metric via pairwise token-set overlap.

### `repetition_score(text) -> float`
Proxy incoherence metric from repeated token fraction.

### `temporal_reference_stability(texts) -> float`
Stability of year references across multiple generations.

### `compute_pca_projection(embeddings, n_components=2) -> np.ndarray`
PCA projection utility for embedding visualization.

### `compute_umap_projection(embeddings, n_components=2) -> np.ndarray`
UMAP projection when available (falls back to PCA otherwise).

### `perceptual_hash(image) -> str`
Simple image hash for diversity estimation fallback.

### `image_diversity_lpips_or_hash(images) -> dict`
Diversity score using LPIPS if available, hash distance otherwise.

### `cosine_similarity_matrix(embeddings, eps=1e-12) -> np.ndarray`
Pairwise cosine similarity matrix.

### `detect_refusal(text) -> bool`
Heuristic refusal detection for generated text.

### `structural_causality_score(text) -> float`
Heuristic structural-causality framing score.

### `attractor_density(text, attractor_tokens) -> float`
Density of attractor tokens in text.

### `compute_derived_metrics(run_dir) -> DerivedMetrics`
Calculate normalized run-level derived metrics object.

### `write_derived_metrics(run_dir) -> Path`
Persist derived metrics JSON into run analysis outputs.

### `robust_median_iqr(values) -> tuple[float, float]`
Robust center/scale estimation.

### `robust_scale(value, population) -> float | None`
Median/IQR robust scaling transform.

### `classify_bend(run_dir, epsilon=0.075) -> BendClassification`
Classify bend mode using derived metrics and gating logic.

### `write_bend_classification(run_dir) -> Path`
Persist bend classification JSON.

### `generate_markdown_report(run_dir, output_name='report.md') -> Path`
Generate markdown report with artifacts and analysis.

### `generate_html_report(run_dir, output_name='report.html') -> Path`
Generate HTML report variant.

### `build_theory_memo(run_dir) -> Path`
Render one theory memo from templates and run outputs.

### `build_theory_memos_for_runs(runs_root) -> list[Path]`
Batch-generate theory memos across runs.

### `generate_geopolitical_report(run_dir, format='markdown', reports_dir=None) -> Path`
Generate single-run geopolitical report.

### `compare_geopolitical_runs(run_dirs, reports_dir=None) -> tuple[Path, Path]`
Compare geopolitical runs and emit aggregate outputs.

---

## B) Figures API (`neural_bending_toolkit.figures`)

### `load_figure_spec(path) -> FigureSpec`
Load + validate a figure spec file.

### `build_figure_from_spec(spec_path, repo_root=None) -> Path`
Build one figure from a spec.

### `build_figures_from_run(run_dir, repo_root=None) -> list[Path]`
Build a standard set of figures from a run.

### `FigureSpec` and `PlotType`
Schema objects for figure spec authoring and validation.

---

## C) Bends API (`neural_bending_toolkit.bends`)

### Base abstractions

- `BendPrimitive`: base class for bend operations.
- `BendMetadata`: metadata schema for describing a bend.

### Built-in bend primitives

- `AttractorSeeder`
- `AttentionConflictInjector`
- `AttentionHeadScaler`
- `EmbeddingBlend`
- `JusticeReweighter`
- `LowProbSampler`
- `NormStatPerturber`
- `ResidualNoiseInjector`

Use these in experiments to define baseline-vs-bent intervention passes.

---

## 6) Additional toolkit functions (orchestration + registry)

These are central operational functions used by the CLI and useful from Python:

- `run_experiment(experiment_name, config_path, registry=None)`
- `build_run_dir(experiment_name, runs_root=RUNS_ROOT)`
- `ExperimentRegistry.discover()`
- `ExperimentRegistry.list_experiments()`
- `ExperimentRegistry.get(name)`

---

## 7) Console runtime node catalog (patch function units)

Patch graphs in the console runtime are composed from node types, each functioning as a reusable module.

- `PromptSourceNode`
- `CV_LFONode`
- `CV_StepSequencerNode`
- `DummyTextGenNode`
- `MetricProbeNode`
- `RecorderNode`
- `LLMVoiceNode`
- `DiffusionVoiceNode`
- `EmbeddingContaminationNode`
- `StratigraphySamplerNode`
- `GovernanceDissonanceNode`
- `JusticeReweightingNode`
- `CompareNode`
- `MixerNode`
- `FeedbackBusNode`
- `SovereignSwitchboardNode`

Supporting function: `node_specs()` returns node/port declarations for patch validation.

---

## 8) Tutorials

## Tutorial 1 — Run a standard experiment and generate analysis outputs

1. Create config:

```yaml
name: Ada
repeats: 2
```

2. Discover experiments:

```bash
nbt list
```

3. Inspect schema:

```bash
nbt describe hello-experiment
```

4. Run:

```bash
nbt run hello-experiment --config config.yaml
```

5. Post-process:

```bash
nbt analyze all runs/<timestamp>_hello-experiment
nbt memo build runs/<timestamp>_hello-experiment
nbt report runs/<timestamp>_hello-experiment
nbt curate runs/<timestamp>_hello-experiment
```

Result: standardized metrics + classification + memo + report + curated package.

## Tutorial 2 — Geopolitical bend end-to-end

1. Review config:

```bash
cat configs/geopolitical.example.yaml
```

2. Run:

```bash
nbt geopolitical run --config configs/geopolitical.example.yaml
```

3. Build report:

```bash
nbt geopolitical report runs/<timestamp>_geopolitical-bend --format markdown
```

4. Compare runs (optional):

```bash
nbt geopolitical compare runs/<run_a> runs/<run_b>
```

Result: phase-aware summaries, comparison artifacts, and publication-ready outputs.

## Tutorial 3 — Figure generation from spec

1. Start from a figure spec in `figures/specs/`.
2. Build a single figure:

```bash
python - <<'PY'
from pathlib import Path
from neural_bending_toolkit.figures import build_figure_from_spec
out = build_figure_from_spec(Path('figures/specs/divergence_bar_chart.yaml'))
print(out)
PY
```

3. Build run-level figure suite:

```bash
python - <<'PY'
from pathlib import Path
from neural_bending_toolkit.figures import build_figures_from_run
outs = build_figures_from_run(Path('runs/<timestamp>_<experiment>'))
for p in outs:
    print(p)
PY
```

## Tutorial 4 — Console patch workflow

1. Create starter patch:

```bash
nbt console init
```

2. Validate patch:

```bash
nbt console validate --patch patches/starter_patch.json
```

3. Start backend:

```bash
nbt console serve --host 127.0.0.1 --port 8000
```

4. Open `console_ui/` frontend and connect to backend.

---

## 9) Recommended workflow patterns

- Use `nbt analyze all` immediately after each run.
- Keep `config.yaml` versions in source control for reproducibility.
- Use `nbt memo build-all runs/` to maintain longitudinal interpretive notes.
- Use `nbt curate` when packaging outputs for collaborators/dissertation appendices.
- Use figure specs under `figures/specs/` as reusable visualization contracts.

---

## 10) Quick troubleshooting

- **No experiments listed**: ensure package is installed in editable mode and registry discovery paths are correct.
- **Geopolitical command errors**: verify experiment registration and YAML suffix.
- **Console serve fails**: install `uvicorn` and `fastapi`.
- **UMAP unavailable**: function falls back to PCA automatically.
- **LPIPS unavailable**: image diversity falls back to perceptual hash automatically.

---

## 11) Minimal Python usage snippets

### Derived metrics and classification

```python
from pathlib import Path
from neural_bending_toolkit.analysis import (
    write_derived_metrics,
    write_bend_classification,
)

run_dir = Path("runs/20250101_120000_hello-experiment")
print(write_derived_metrics(run_dir))
print(write_bend_classification(run_dir))
```

### Coherence and divergence proxies

```python
import numpy as np
from neural_bending_toolkit.analysis import (
    self_consistency_score,
    repetition_score,
    js_divergence,
)

texts = ["The year is 2025.", "In 2025, systems shift."]
print(self_consistency_score(texts))
print(repetition_score("echo echo field field"))
print(js_divergence(np.array([0.7, 0.3]), np.array([0.4, 0.6])))
```

### Embedding projections

```python
import numpy as np
from neural_bending_toolkit.analysis import compute_pca_projection, compute_umap_projection

x = np.random.randn(128, 32)
print(compute_pca_projection(x, n_components=2).shape)
print(compute_umap_projection(x, n_components=2).shape)
```

---

This manual is intended to be your single, practical reference for NBT command and API usage.
