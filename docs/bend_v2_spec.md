# Bend v2 Specification

Bend v2 defines an execution-layer interface for **architectural metastability interventions** in NBT runs. It is designed to be adapter-agnostic while still matching current NBT seams (for example, the Diffusers adapter embedding hook and cross-attention hook seam in `models/diffusion_diffusers.py`).

A Bend v2 declaration has a four-part contract:

1. **Site** — where in model execution the bend acts.
2. **Actuator** — what variable or signal is perturbed.
3. **Schedule** — when/how the perturbation is applied.
4. **Trace** — what evidence is logged to show metastability changed.

---

## 1) Site

`site` identifies the intervention locus with enough precision to replay or compare runs.

Suggested fields:

- `adapter`: NBT adapter family (`llm_hf`, `diffusion_diffusers`, `gan_stylegan`, `audio_gen`)
- `module`: named submodule or seam (`cross_attn`, `unet.mid_block.attentions.0`, `transformer.h.11.attn`)
- `head`: optional attention head selector (`all`, index, list)
- `layer`: optional layer selector (single index, range, explicit list)
- `timestep`: execution coordinate (token position, denoising timestep, frame index)

Examples:

- Diffusion cross-attention seam at selected denoising timesteps.
- LLM residual stream at a layer band for token positions matching a predicate.
- Audio transformer attention heads only during a time-windowed segment.

---

## 2) Actuator

`actuator` names the perturbed quantity and defines the operation.

Common actuator targets:

- Attention internals: `Q`, `K`, `V`, `attention_probs`
- Normalization parameters: norm `gain` / `bias`
- Residual pathways: additive or gated residual injection
- Embedding pathways: embedding projection or embedding mix
- Adapter-specific latent/state tensors exposed at hook seams

Common operations:

- `add`: additive delta
- `scale`: multiplicative scaling
- `mix`: blend between baseline and alternate tensor
- `replace`: hard substitution
- `mask`: selective suppression/amplification

Notes:

- For `diffusion_diffusers`, actuator targets should map to existing embedding and cross-attn intervention surfaces.
- Actuators should be serializable so configs can be versioned and audited.

---

## 3) Schedule

`schedule` defines temporal logic of intervention.

Supported schedule patterns (spec-level):

- **Window**: active between start/end timesteps
- **Ramp**: linearly or nonlinearly changing strength over interval
- **Pulse**: periodic or one-shot spikes
- **Conditional trigger**: activate when a runtime predicate holds

Typical controls:

- `strength`: base coefficient
- `warmup` / `cooldown`
- `frequency` / `duty_cycle` for pulses
- `condition` expression for trigger-based activation

Interpretation rule:

- If multiple schedule clauses overlap, combine via explicit policy (`sum`, `max`, `last_write_wins`).

---

## 4) Trace

`trace` defines evidence that the bend altered metastability (not just output surface form).

Recommended trace metrics:

- Attention entropy shifts (per head/per layer)
- Activation norm changes (L2 / RMS before vs after bend)
- Latent delta magnitude (`||x_bent - x_base||`)
- Recovery-after-shock metrics (time-to-baseline, overshoot, settling behavior)
- Optional task-level side metrics for context (quality, diversity, refusal rate, etc.)

Trace requirements:

- Log baseline and bent values for the same execution coordinates.
- Attach sufficient metadata (`site`, schedule phase, seed, prompt/input hash) to reproduce comparisons.
- Persist to standard NBT run outputs (`metrics.jsonl`, artifacts) so runs remain reportable.

---

## Bend v2 YAML example (proposed)

```yaml
bend_v2:
  id: "diffusion.cross_attn_entropy_probe.v1"
  site:
    adapter: "diffusion_diffusers"
    module: "cross_attn"
    layer: [3, 4, 5]
    head: "all"
    timestep:
      kind: "denoise_step"
      window: [120, 420]
  actuator:
    target: "attention_probs"
    op: "mix"
    params:
      source: "alternate_prompt_attention"
      alpha: 0.25
  schedule:
    policy: "sum"
    phases:
      - kind: "ramp"
        start: 120
        end: 220
        strength_start: 0.0
        strength_end: 1.0
      - kind: "window"
        start: 220
        end: 360
        strength: 1.0
      - kind: "pulse"
        start: 360
        end: 420
        frequency: 0.2
        duty_cycle: 0.3
        strength: 0.7
      - kind: "conditional"
        condition: "attn_entropy(layer=4) < 1.2"
        strength: 0.4
  trace:
    metrics:
      - "attention_entropy"
      - "activation_norm_l2"
      - "latent_delta_magnitude"
      - "recovery_after_shock"
    compare_to: "baseline"
    log_to:
      - "metrics.jsonl"
      - "artifacts/attn_maps"
```

This example is declarative documentation and does not imply full runtime support yet.

---

## Taxonomy mapping: Stratigraphy vs Proto-bends vs Architectural bends

Use this mapping to label experiments consistently in NBT:

| Category | Primary question | Intervention locus | Typical method | Bend v2 relation |
|---|---|---|---|---|
| **Stratigraphy (regime cartography)** | Which behavioral regimes appear under sampling/config pressure? | Sampling/inference controls external to execution internals | Temperature/truncation sweeps, prompt-set cartography | Adjacent but distinct; not a Bend v2 intervention |
| **Proto-bends (representation perturbation)** | How do representational states shift when internal vectors are perturbed? | Embeddings/latents/feature vectors | Embedding contamination, latent blending, representation edits | Can be expressed in Bend v2 when perturbation occurs at a defined site+schedule |
| **Architectural bends (stabilization perturbation)** | How does model stability change when architectural flow is perturbed? | Attention, residual, norm, routing pathways during execution | Q/K/V edits, attention-prob steering, residual injections, norm gain/bias modulation | Core target of Bend v2 |

Practical labeling recommendation:

- Add `taxonomy_label` metadata to experiments/runs with one of:
  - `stratigraphy`
  - `proto_bend`
  - `architectural_bend`
- If an experiment combines sweep + intervention, mark primary label by causal emphasis and add secondary notes in run metadata.

---

## Non-goals

- Bend v2 does not replace sampler studies; it complements them.
- Bend v2 does not mandate a single adapter implementation strategy.
- Bend v2 spec compliance is possible incrementally per adapter and per hook seam.
