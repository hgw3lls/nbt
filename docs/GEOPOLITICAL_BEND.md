# Geopolitical Bend

## Conceptual framing

The Geopolitical Bend suite examines governance concepts, contradiction stress, and justice attractor dynamics as a way to surface model priors around policy, causality, and refusal behavior.

## Run commands

```bash
nbt geopolitical describe
nbt geopolitical run --config configs/geopolitical.example.yaml
```

## Reporting commands

```bash
nbt geopolitical report runs/<timestamp>_geopolitical-bend --format markdown
nbt geopolitical compare runs/<run1> runs/<run2>
```

## Outputs

- Phase 1: embedding/similarity diagnostics
- Phase 2: contradictory-prompt stress metrics
- Phase 3: baseline-vs-attractor comparisons
- Reports: timestamped markdown/html + figure outputs under `reports/`
