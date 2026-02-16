# Runs and Artifacts

## Run directory structure

Each run is created under:

```text
runs/<timestamp>_<experiment>/
```

Common files:

- `config.yaml`: resolved run configuration.
- `metrics.jsonl`: structured metric records.
- `events.log`: human-readable logs.
- `events.jsonl`: structured event stream.
- `artifacts/`: saved outputs (images, text, arrays, audio, reports).

## `metrics.jsonl` shape

Each line is JSON with keys typically including:

- `step`
- `metric_name`
- `value`
- `metadata`

This format supports downstream plotting, comparison tables, and report generation.

## Artifact strategy

Artifacts should be organized by experiment phase and media type. For geopolitical runs, phase artifacts live under:

```text
artifacts/geopolitical/phase_1_ontology_mapping/
artifacts/geopolitical/phase_2_governance_dissonance/
artifacts/geopolitical/phase_3_justice_attractors/
```

Keep artifact naming stable so figures can be cited directly in dissertation drafts.
