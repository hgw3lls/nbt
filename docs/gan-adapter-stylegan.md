# GAN Adapter (StyleGAN)

The `models/gan_stylegan.py` adapter wraps local StyleGAN2/3 generators for controlled latent-space experiments.

## Install

```bash
pip install -e .[dev,gan]
```

## Loading local checkpoints

`StyleGANAdapter` supports local checkpoints saved as:

- a generator module directly
- a dictionary containing `generator`
- a dictionary containing `G_ema`

## Latent utilities

- `sample_latents(n, seed=...)`
- `linear_interpolate(a, b, t)`
- `slerp(a, b, t)`
- `traverse_direction(z, direction, scale=..., toward=True/False)`

## Experiment

`gan-stratigraphy-edges` compares low vs high truncation sampling, logs diversity metrics, and saves montage artifacts.
