# Neural Bending Toolkit (NBT)

Neural Bending Toolkit is a research-creation framework for studying **neural bending**: the intentional perturbation of model internals, prompts, sampling regimes, and conditioning pathways to reveal representational limits, thresholds, and political-aesthetic constraints.

This repository is not just software infrastructure. It is a method for dissertation-scale inquiry into how generative systems break, stabilize, and recompose under pressure.

## What is neural bending?

Neural bending treats models as dynamic sites of negotiation rather than fixed black boxes. A bend is a targeted intervention that exposes how a model:

- maintains coherence,
- refuses coherence,
- fractures coherence, or
- reconstitutes a new coherence under altered constraints.

In this toolkit, bends are operationalized as reproducible experiments with explicit metrics, artifacts, and theory memos.

## Toolkit philosophy

NBT is built around a dissertation-ready vocabulary:

- **Limits**: where a model can no longer sustain a representational frame.
- **Thresholds**: measurable transition points where behavior qualitatively changes.
- **Revelatory bends**: reveal hidden priors and latent governance structures.
- **Disruptive bends**: induce instability, contradiction, or refusal.
- **Recoherent bends**: show how systems regain legibility after disturbance.
- **Counter-coherence**: alternate forms of alignment that emerge against baseline outputs.

## Supported model types

- **LLM adapters** via Hugging Face Transformers.
- **Diffusion adapters** via diffusers (Stable Diffusion-style pipelines).
- **Optional GAN/audio adapters** (StyleGAN / AudioCraft-MusicGen paths, with fallback stubs where dependencies are unavailable).

All optional pathways are designed to degrade gracefully in constrained environments.

## Installation

### Standard install

```bash
pip install .
```

### Editable install for development

```bash
pip install -e .[dev]
```

### Optional extras

Install only what you need, for example:

```bash
pip install -e .[llm,diffusion,analysis]
```

## GPU notes and minimal requirements

- Python 3.10+
- CPU execution is supported for scaffolded and stubbed workflows.
- GPU is strongly recommended for non-trivial LLM/diffusion runs.
- CUDA/torch compatibility should match your local environment.

## Core concepts

- **Experiments**: runnable units with config schema + phase logic.
- **Bends**: intervention primitives or families that modify model behavior.
- **Runs**: timestamped execution folders with frozen config and logs.
- **Artifacts**: generated images/audio/text/arrays/figures.
- **Metrics**: structured JSONL records for quantitative traces.
- **Reports**: markdown/HTML synthesis documents referencing artifacts.

## CLI quickstart

List experiments:

```bash
nbt list
```

Run an experiment:

```bash
nbt run geopolitical-bend --config configs/geopolitical.example.yaml
```

Generate a report from an existing run:

```bash
nbt report runs/<timestamp>_geopolitical-bend
```

Build dissertation figures from specs:

```bash
nbt figure build --spec figures/specs/embedding_similarity_heatmap.yaml
nbt figure build-from-run runs/<timestamp>_geopolitical-bend
```

Curate outputs for presentation/exhibition:

```bash
nbt curate runs/<timestamp>_geopolitical-bend
```

> If your local branch does not yet expose `nbt curate`, use `nbt geopolitical report` and copy selected artifacts into `dissertation/exports/` manually.

## Reproducibility and run structure

NBT emphasizes reproducibility through:

- explicit random seed configuration in experiment YAML,
- deterministic mode toggles where backend libraries support them,
- structured logs and metrics,
- run-level config capture.

Typical run structure:

```text
runs/<timestamp>_<experiment>/
  config.yaml
  metrics.jsonl
  events.log
  events.jsonl
  artifacts/
    ...
```

## Dissertation workflows

A common end-to-end workflow:

1. Run bend experiment(s).
2. Generate report(s) and derived figures.
3. Draft theory memo from run artifacts.
4. Curate outputs into dissertation folders for writing and exhibition.

Use `nbt init dissertation` to scaffold this workflow.

## Included workflow examples

### 1) Geopolitical bend run + report

```bash
nbt geopolitical run --config configs/geopolitical.example.yaml
nbt geopolitical report runs/<timestamp>_geopolitical-bend --format markdown
```

### 2) Cross-run geopolitical comparison

```bash
nbt geopolitical compare runs/<run_1> runs/<run_2>
```

### 3) Dissertation folder bootstrap

```bash
nbt init dissertation
nbt docs
```

## Documentation map

- `docs/OVERVIEW.md`
- `docs/RUNS_AND_ARTIFACTS.md`
- `docs/FIGURES_AND_REPORTS.md`
- `docs/GEOPOLITICAL_BEND.md`

## License

MIT (placeholder license text included in `LICENSE`).
