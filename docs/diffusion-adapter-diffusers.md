# Diffusion Adapter (Diffusers)

The `models/diffusion_diffusers.py` adapter wraps Stable Diffusion-style pipelines and exposes hooks for embedding/cross-attention intervention.

## Install

```bash
pip install -e .[dev,diffusion]
```

## CPU-friendly default model

For local CPU experiments use:

- `hf-internal-testing/tiny-stable-diffusion-pipe`

## Hook points

- **Embedding hook**: intercept/modify CLIP text embeddings before denoising.
- **Cross-attention hook**: intercept/modify query, key, or attention weights at denoising steps.

## Artifacts

The adapter can save:

- generated images (`.png`)
- intermediate attention heatmaps (`.npy`)

## Example experiment

Use `embedding-contamination-diffusion` to blend base and contaminant text embeddings and compare image outputs + attention maps.
