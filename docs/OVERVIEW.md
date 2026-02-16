# NBT Overview

## Conceptual framing

Neural Bending Toolkit (NBT) operationalizes neural bending as a research-creation method for probing representational limits and coherence regimes in generative models.

NBT supports bends that are:

- revelatory,
- disruptive,
- recoherent,
- counter-coherent.

## Architecture at a glance

- `src/neural_bending_toolkit/experiment.py`: experiment and run abstractions.
- `src/neural_bending_toolkit/bends/`: bend families and experiment-specific bend implementations.
- `src/neural_bending_toolkit/models/`: model adapters (LLM, diffusion, GAN, audio, and metadata helpers).
- `src/neural_bending_toolkit/analysis/`: metrics, report generation, and geopolitical analysis utilities.
- `src/neural_bending_toolkit/cli.py`: Typer CLI commands.

## Core workflow

1. Configure an experiment via YAML.
2. Run with `nbt run ...` or experiment-specific commands.
3. Inspect run artifacts and metrics.
4. Generate reports and figures.
5. Curate outputs for dissertation writing and exhibition pathways.
