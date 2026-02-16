# Runs, Metrics, Artifacts

Neural Bending Toolkit creates a run directory per execution:

- `runs/<timestamp>_<experiment>/config.yaml`
- `runs/<timestamp>_<experiment>/events.log`
- `runs/<timestamp>_<experiment>/events.jsonl`
- `runs/<timestamp>_<experiment>/metrics.jsonl`
- `runs/<timestamp>_<experiment>/artifacts/`

## Logging

`events.log` contains human-readable logs.

`events.jsonl` contains structured JSON records with:

- `timestamp`
- `level`
- `event`
- `payload`

## Metrics

`metrics.jsonl` contains one metric per line with fields:

- `timestamp`
- `step`
- `metric_name`
- `value`
- `metadata`

## Artifacts

Use `RunContext` helpers:

- `save_text_artifact(filename, text)`
- `save_image_artifact(filename, image)`
- `save_numpy_artifact(filename, array)`
- `save_torch_artifact(filename, tensor)` (requires torch)

Image arrays require Pillow (`pip install .[images]`).

## Weights & Biases integration

W&B logging is optional. Enable via environment variable:

```bash
export NBT_ENABLE_WANDB=true
```

Install optional dependency:

```bash
pip install .[wandb]
```

## Intervention snapshots

Experiments can log pre/post intervention snapshots:

- `pre_intervention_snapshot(name, data)`
- `post_intervention_snapshot(name, data)`

These emit structured events into `events.jsonl`.
