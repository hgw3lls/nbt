# Getting Started

Install the toolkit in editable mode:

```bash
pip install -e .[dev]
```

Create a config file:

```yaml
name: Ada
repeats: 2
```

Then run:

```bash
nbt list
nbt describe hello-experiment
nbt run hello-experiment --config config.yaml
```

## CPU-friendly local LLM config

Use the small default model (`sshleifer/tiny-gpt2`) for CPU workflows:

```yaml
model_name: sshleifer/tiny-gpt2
prompt: "The neural manifold is"
max_new_tokens: 16
temperatures: [0.8, 1.0]
top_ps: [0.9, 1.0]
top_ks: [0, 40]
```

Install optional dependencies before running this experiment:

```bash
pip install -e .[llm]
```


## CPU-friendly diffusion config

```yaml
model_id: hf-internal-testing/tiny-stable-diffusion-pipe
base_prompt: "a clean laboratory bench"
contaminant_prompt: "a chaotic graffiti wall"
contamination_alpha: 0.25
num_inference_steps: 10
guidance_scale: 7.0
seed: 7
```

Install optional dependencies:

```bash
pip install -e .[diffusion]
```


## GAN edge-stratigraphy config

```yaml
checkpoint_path: /path/to/stylegan_checkpoint.pt
latent_dim: 512
n_samples: 8
low_truncation: 0.3
high_truncation: 1.4
seed: 17
```

Install optional dependencies:

```bash
pip install -e .[gan]
```


## Audio drift config

```yaml
prompt: "A minimalist ambient synth line"
duration_s: 3.0
baseline_conditioning_scale: 1.0
drift_conditioning_scale: 1.4
baseline_attention_scale: 1.0
drift_attention_scale: 1.6
model_name: facebook/musicgen-small
seed: 13
```

Install optional dependencies:

```bash
pip install -e .[audio]
```
