# Console Real Models + Bend Nodes

This document describes the Neural Bending Console support for real model adapters and CI-safe mocks.

## Adapter policy

The console now supports two adapter-backed voice nodes:

- `LLMVoiceNode` (HF causal LM wrapper)
- `DiffusionVoiceNode` (diffusers wrapper)

To keep CI safe (no model downloads), console adapters default to mocks.

Set:

```bash
export NBT_CONSOLE_USE_MOCK_ADAPTERS=false
```

to use real adapters.

## New nodes

- `LLMVoiceNode`
  - Inputs: `prompt`, `temp_cv`, `top_p_cv`, `top_k_cv`
  - Outputs: `text`, `metric`
- `DiffusionVoiceNode`
  - Inputs: `prompt`, `guidance_cv`, `embedding`
  - Outputs: `image_path`, `metric`
- `EmbeddingContaminationNode`
  - Inputs: `embedding_a`, `embedding_b`
  - Output: `embedding`
- `StratigraphySamplerNode`
  - Input: `cv`
  - Output: `cv` sampler control dict
- `GovernanceDissonanceNode`
  - Inputs: `text`, `embedding_a`, `embedding_b`, `cv`
  - Outputs: `text`, `embedding`, `cv`
- `JusticeReweightingNode`
  - Inputs: `text`, `embedding`
  - Outputs: `text`, `embedding`, `metric`
- `CompareNode`
  - Inputs: `baseline_text`, `bent_text`, `baseline_image`, `bent_image`
  - Output: `metric`

## Example patches

- `patches/llm_ab_compare.json`
- `patches/diffusion_ab_compare.json`

## Running

```bash
nbt console serve --host 127.0.0.1 --port 8000
```

With UI:

```bash
cd console_ui
npm run dev
```
