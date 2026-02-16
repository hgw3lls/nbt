# CLI Reference

## `nbt list`

List all discovered experiments.

## `nbt describe <experiment>`

Show description and schema for an experiment.

## `nbt run <experiment> --config <path.yaml>`

Run an experiment with a YAML config file and write outputs in `runs/`.

## LLM sampling stratigraphy example

```bash
nbt describe llm-sampling-stratigraphy
nbt run llm-sampling-stratigraphy --config llm_config.yaml
```

## Diffusion embedding contamination example

```bash
nbt describe embedding-contamination-diffusion
nbt run embedding-contamination-diffusion --config diffusion_config.yaml
```

## GAN stratigraphy edges example

```bash
nbt describe gan-stratigraphy-edges
nbt run gan-stratigraphy-edges --config gan_config.yaml
```

## Audio inter-head drift example

```bash
nbt describe audio-inter-head-drift
nbt run audio-inter-head-drift --config audio_config.yaml
```

## Dissertation bend families

```bash
nbt list
nbt describe family-embedding-contamination
nbt run family-embedding-contamination --config family.yaml
```

## Run report

```bash
nbt report runs/<timestamp>_<experiment>
```

## Geopolitical Bend

```bash
nbt geopolitical describe
nbt geopolitical run --config configs/geopolitical.example.yaml
nbt geopolitical report runs/<timestamp>_geopolitical-bend
nbt geopolitical compare runs/<run1> runs/<run2>
```
