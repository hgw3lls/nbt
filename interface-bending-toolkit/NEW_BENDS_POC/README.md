# NEW_BENDS_POC Field Guide (LIMITS OF CTRL Edition)

This folder collects prototype "neural bend" experiments framed through the **LIMITS OF CTRL** theory: instead of striving for perfectly controlled generation, each script intentionally **mis-wires** a transformer or CLIP-like model to surface failure modes, bias pathways, and emergent behaviors. The experiments borrow from circuit bending—lifting traces, swapping gain stages, and injecting noise—to expose how the model distributes control across embeddings, attention, normalization, memory, and multimodal fusion.

Use this guide to understand what each script does, how it maps to LIMITS OF CTRL, and what to look for in the outputs. Each script logs runs to a subfolder of `runs/` and prints lightweight diagnostics (e.g., cosine drifts, gain maps, key noise traces) to help interpret how the bend changed the system.

## How to read the results
- **Baseline vs. bent**: Most scripts generate a baseline continuation and a bent one; the gap between them shows the controllability boundary for that subsystem.
- **Dry/wet or mix controls**: Many bends expose a `mix` parameter to blend original and bent signals—treat it like an attenuation pot to feel when control gives way to drift.
- **Diagnostics**: Cosine shifts, token hit counts, activation deltas, or ranking swaps show where the model ceded control. Look for sudden spikes or inversions; they mark the fault lines the LIMITS OF CTRL theory is interested in.
- **Interpretation**: Expect side effects—nonsense tokens, misplaced emphasis, or unstable generations. These are *features*, not bugs: they map the edges of controllability.

## Script index
Below, each script is paired with a short description, its circuit-bending metaphor, and what to expect.

### Embedding and semantic bends
- **`embedding_bend.py`** – Cross-wires token embeddings via drift or inversion to re-route conceptual proximity; watch cosine similarity changes and the oscilloscope-style drift graph to see ontology bending in action.
- **`embedding_neighborhood_drift.py`** – Nudges clusters of nearby embeddings toward alternative neighborhoods, surfacing how local geometry affects lexical recall and where semantic control fractures.
- **`semantic_inversion.py`** – Mirrors selected embedding directions to test how fragile meaning is when conceptual axes flip, a direct probe of LIMITS OF CTRL around representational symmetry.
- **`latent_bend.py`** – Injects an attractor vector into hidden states to bias the latent bus; the resulting pull shows how little force is needed before narratives slip toward the attractor.
- **`latent_attractor_reorientation.py`** – Rotates attractor directions mid-stream to reveal inertia and hysteresis in the latent space, highlighting delayed or partial compliance with control signals.

### Attention and relevance bends
- **`attention_bend.py`** – Adds boost/suppress masks to logits to over-amplify chosen tokens, simulating patched relevance buses; expect gain maps that show when amplification overwhelms the model’s native salience.
- **`attention_head_disalign.py`** – Forces two head groups to disagree by mixing competing logit biases; per-step A/B probability bars show how divergent guidance produces unstable yet telling continuations.
- **`divergent_heads_demo.py`** – Demonstrates head-level disagreement with simplified hooks, illustrating the LIMITS OF CTRL concept that coordination across heads is brittle.
- **`subaltern_attention_boost.py`** – Elevates marginalized vocabulary within attention-like biases to test how much systemic headroom the model affords subaltern terms before fluency degrades.

### Residual and activation bends
- **`residual_bend.py`** – Applies bottleneck or relational projections to residual streams, acting as choke or feedback patches; singular values and cosine drifts show where expressivity collapses or reorients.
- **`residual_bottleneck_demo.py`** – Focused bottleneck demo that strips residual capacity to expose minimal grammar pathways.
- **`relational_residual_demo.py`** – Adds relational direction vectors to the residual stream, foregrounding communal or structural narratives and revealing resistance points.
- **`middle_layer_shear.py`** – Shears mid-layer activations by mixing left/right halves, akin to twisting PCB traces; layer-wise L2 deltas quantify how geometry skew translates into thematic drift.
- **`gradient_freeze_reroute.py`** – Freezes most parameters and fine-tunes only a chosen shard, rerouting feedback through a tiny patch; compare baseline vs. locally adapted outputs to see how little plasticity can still bend behavior.

### Normalization and stability bends
- **`layernorm_bend.py`** – Biases logits with care-centered similarity or variance noise, shifting the model’s “voltage reference” of normality; similarity histograms mark how stability tilts toward care vocabularies.
- **`layernorm_inversion.py`** – Swaps LayerNorm weights between early/late layers to invert gain staging; weight-distance readouts show where stabilization roles flip and coherence frays.
- **`care_norm_demo.py`** – Lightweight illustration of norm shifts toward care/harm reduction vocab, highlighting norm as a lever in LIMITS OF CTRL.
- **`layernorm_variance.py` / `layernorm_recenter.py`** – Variance jittering and center-shifting variants (if present) to test sensitivity of normalization anchors.

### Positional and temporal bends
- **`positional_bend.py`** – Scrambles clauses or anchors prompts with historical phrases to miswire positional expectations; before/after listings reveal how the model repairs temporal disruptions.
- **`temporal_scramble_demo.py`** – Minimal temporal scramble example showing how small order changes ripple through generation.
- **`justice_temporal_anchor.py`** – Inserts justice-oriented anchors to bias temporal framing, probing how persistent context injections reshape narrative timelines.

### Multimodal and cross-channel bends
- **`multimodal_bend.py`** – Bends CLIP alignments via subaltern directions or mismatched pairing; top-k rank shifts expose how fragile text–image coherence is when alignment is patched.
- **`clip_mismatch_injection.py`** – Forces systematic text/image mismatches to audit how confidently the system asserts wrong pairings—an embodiment of control limits in multimodal fusion.
- **`cross_modal_reweighting.py`** – Reweights cross-modal scores to privilege chosen vocabularies, revealing leverage points where one modality can dominate.
- **`modality_parallax.py`** – Rotates text embeddings in a random plane to create parallax errors; before/after rankings map the angle at which fusion loses grip.

### Structural and ideological bends
- **`structural_drift.py`** – Boosts structural/systemic vocabulary while suppressing individualizing terms; token hit counts quantify ideological drift and show where language of systems overtakes blame narratives.
- **`structural_drift_demo.py`** – Smaller demo variant emphasizing the same systemic tilt with fewer controls.

### Memory and sampling bends
- **`kv_memory_drift.py`** – Adds noise to cached keys (and optionally values) between decoding steps, corrupting addresses more than content; noise traces chart how retrieval misalignment derails coherence.
- **`synthetic_perplexity_injection.py`** – Inverts probability mass to favor low-prob tokens, like feeding a lying VU meter to the sampler; per-step rank traces show when aesthetic chaos overtakes likelihood.

### Positional/anchor hybrids and other prompts
- **`positional_anchor.py` / `positional_scramble.py`** – If present, simplified anchor/scramble utilities paralleling `positional_bend.py` for quick tests.

### Orchestration and utilities
- **`run_all_bends.py`** – Orchestrator that runs multiple bends with a shared prompt, aggregates logs, and emits a Markdown report—useful for comparative LIMITS OF CTRL figure sets.
- **`requirements.txt`** – Minimal dependency pinning for the prototypes; most scripts rely on `transformers`, `torch`, and optionally `open_clip`/Pillow for multimodal bends.

## Interpreting through LIMITS OF CTRL
Each bend demonstrates a different **fault line of controllability**:
- **Geometry vs. meaning**: Embedding shears and inversions (e.g., `embedding_bend.py`, `middle_layer_shear.py`) show how small geometric changes rewire meaning.
- **Coordination vs. discord**: Head disagreement and structural drifts (`attention_head_disalign.py`, `divergent_heads_demo.py`) reveal that coherent output depends on fragile agreement among submodules.
- **Stability vs. sensitivity**: LayerNorm and residual bends (`layernorm_inversion.py`, `residual_bend.py`) test how stabilization and bottlenecks enforce or erode control.
- **Memory vs. noise**: KV drift and perplexity injection (`kv_memory_drift.py`, `synthetic_perplexity_injection.py`) expose reliance on clean addresses and truthful uncertainty estimates.
- **Fusion vs. misalignment**: Multimodal parallax and mismatch (`modality_parallax.py`, `multimodal_bend.py`) highlight how precise cross-modal welds must be to sustain coherence.

Collectively, these scripts operationalize LIMITS OF CTRL by *bending* rather than optimizing: they map where gentle pushes cause disproportionate drift, making the hidden control circuitry of generative models visible.
