# Neural Bending Toolkit

A production-grade Python toolkit scaffold for experiments and command-line workflows.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
nbt --help
```

## Experiment framework

Experiments are discovered from:

- Python entry points in the `nbt.experiments` group
- Built-in module scanning under `neural_bending_toolkit.experiments`

Each run writes to:

- `runs/<timestamp>_<experiment>/config.yaml`
- `runs/<timestamp>_<experiment>/events.log`
- `runs/<timestamp>_<experiment>/events.jsonl`
- `runs/<timestamp>_<experiment>/metrics.jsonl`
- `runs/<timestamp>_<experiment>/artifacts/`

## CLI commands

- `nbt list`
- `nbt describe <experiment>`
- `nbt run <experiment> --config path.yaml`
- `llm-sampling-stratigraphy` experiment available for sampling analysis
- `embedding-contamination-diffusion` experiment for diffusion embedding interventions
- `gan-stratigraphy-edges` experiment for truncation-edge GAN analysis
- `audio-inter-head-drift` experiment for MusicGen/proxy drift analysis

Example config:

```yaml
name: Ada
repeats: 2
```

## Instrumentation

- Structured logging (JSON + human-readable)
- Metrics logger (`step`, `metric_name`, `value`, `metadata`)
- Artifact APIs for text/images/numpy/torch
- Optional Weights & Biases integration via `NBT_ENABLE_WANDB=true`
- Pre/post intervention snapshot hooks

See [Runs, Metrics, Artifacts](docs/runs-metrics-artifacts.md).

## Development

```bash
ruff check .
black --check .
pytest
```

See [`Install.md`](Install.md) for installation workflows, [`docs/`](docs/) for documentation stubs, and [`CHANGELOG.md`](CHANGELOG.md) for release notes.


## LLM adapter

A Hugging Face adapter is available in `models/llm_hf.py` with a CPU-friendly default model: `sshleifer/tiny-gpt2`.
Install with optional extras: `pip install -e ".[llm]"`.

A Diffusers adapter is available in `models/diffusion_diffusers.py` with a CPU-friendly default model: `hf-internal-testing/tiny-stable-diffusion-pipe`.
Install with optional extras: `pip install -e ".[diffusion]"`.

A StyleGAN adapter is available in `models/gan_stylegan.py` for local checkpoints.
Install with optional extras: `pip install -e ".[gan]"`.

An audio adapter is available in `models/audio_gen.py` (MusicGen if available, else proxy).
Install with optional extras: `pip install -e ".[audio]"`.


## Dissertation bend families

Eight bend-family experiments are included (embedding contamination, corpus stratigraphy, inter-head drift, governance dissonance, residual distortion, norm perturbation, justice reweighting, justice attractors).
Each run emits baseline-vs-bent comparisons, metrics + qualitative samples, and a `theory_memo.md`.

### Flagship experiment: governance-shock-recovery-diffusion

Run with:

```bash
nbt run governance-shock-recovery-diffusion -c configs/governance-shock-recovery-diffusion.example.yaml
```

This experiment demonstrates metastability bending by applying Bend v2 **Site · Actuator · Schedule · Trace** interventions across three conditions: `baseline`, `shock`, and `shock_counter`.

- **Site**: diffusion cross-attention (`attn2`/cross-attn layer regex + targeted heads)
- **Actuator**: head gating, attention temperature, and qk rotation
- **Schedule**: mid-window shock with late-window counter intervention
- **Trace**: attention entropy/top-k mass and downstream metastability tags

Key artifacts to inspect:

- `conditions/{baseline,shock,shock_counter}/...`
- `comparisons/metrics_comparison.json`
- `comparisons/comparison_report.json`
- `comparisons/entropy_over_steps.png`
- `summary.json` (metastability + tags)

Generate analysis reports from existing runs with `nbt report <run_dir>`.


Geopolitical suite: `nbt geopolitical describe` and `nbt geopolitical run --config configs/geopolitical.example.yaml`.



## Bend v2: Architectural Metastability Interface

Bend v2 formalizes an execution-layer contract for interventions defined by **Site · Actuator · Schedule · Trace**. See [`docs/bend_v2_spec.md`](docs/bend_v2_spec.md).

Sampler sweeps are stratigraphy; Bend v2 is execution-layer intervention.

## Starter patches

Starter modular patches live under `patches/`, including `patches/starter_8_bends_ab.json`.

```bash
nbt console serve
# then in console_ui: npm run dev and load patches/starter_8_bends_ab.json
```
