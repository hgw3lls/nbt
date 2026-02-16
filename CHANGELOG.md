# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Geopolitical analysis reporting utilities with `nbt geopolitical report` and `nbt geopolitical compare` for timestamped report/compare outputs, plots, and side-by-side summaries.
- Common instrumentation layer with structured and human-readable logging.
- JSONL metrics logger with step/metric/value/metadata schema.
- Artifact saver APIs for text, images, numpy arrays, and torch tensors.
- Optional Weights & Biases integration behind `NBT_ENABLE_WANDB` feature flag.
- Pre/post intervention snapshot hooks in run context.
- Documentation page: "Runs, Metrics, Artifacts."
- Hugging Face causal LM adapter with unified generation and model-signal capture utilities.
- New `llm-sampling-stratigraphy` experiment with top-p/top-k/temperature sweeps and KL-vs-baseline logging.
- LLM adapter documentation including CPU-friendly default model guidance.
- Diffusers Stable Diffusion-style adapter with embedding and cross-attention intervention hooks.
- New `embedding-contamination-diffusion` experiment blending text embeddings and comparing outputs/attention maps.
- Diffusion adapter docs with CPU-friendly default model guidance.
- StyleGAN2/3 adapter with local checkpoint loading, latent interpolation, and direction traversal tools.
- New `gan-stratigraphy-edges` experiment sampling near truncation edges with diversity metrics and montage artifacts.
- StyleGAN adapter documentation with setup and usage guidance.
- Audio adapter with MusicGen backend (when available) and explicit proxy fallback mode.
- New `audio-inter-head-drift` experiment with backend-aware limitations and WAV artifact outputs.
- Audio adapter documentation and CLI/getting-started examples.
- Dissertation bend primitive set with explicit modification targets, safety constraints, and rollback behavior.
- Eight dissertation bend-family experiments with baseline/bent comparisons, metrics, qualitative samples, and theory memos.
- Run analysis utilities for embedding topology, attention entropy/divergence, KL, coherence proxies, and image diversity proxies.
- New `nbt report <run_dir>` command generating markdown reports with artifact citations.
- Full Geopolitical Bend implementation with ontology mapping, governance dissonance stress testing, justice attractor injection, reusable geopolitical analysis utilities, deterministic seeding, and phased artifact outputs.
