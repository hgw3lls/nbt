# LLM Adapter (Hugging Face)

The `models/llm_hf.py` adapter exposes a unified interface for causal LM generation and signal capture.

## Install

Use optional extras:

```bash
pip install -e .[dev,llm]
```

## CPU-friendly default model

For quick local experiments, use:

- `sshleifer/tiny-gpt2`

This model is intentionally small for CPU workflows.

## API overview

- `generate(prompt, **params)` → returns generated text and token-level logprobs (when available)
- `capture_model_signals(...)` → capture hidden states, attention maps, logits for selected layers/positions

## Notes for tests

Do not hardcode model downloads in unit tests. Tests should mock adapter behavior or validate pure utility functions.
