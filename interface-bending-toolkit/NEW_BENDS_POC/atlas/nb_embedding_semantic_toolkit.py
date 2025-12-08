"""
nb_embedding_semantic_toolkit.py

Unified toolkit for:
    - Embedding & Semantic Bends
    - Latent Attractor Bends (mid-layer + final hidden MOD BUS)
    - Theory-friendly logging (axis/mode/bend_name, diffs, similarity meters)
    - Config-driven experiment runner

Includes:

  * Model loaders (MLM / CLM / GPT-2 LMHead)
  * Embedding-level bends:
        - Cluster (directional) drift
        - Pairwise drift
        - Semantic inversion (midpoint reflection)
  * Masked-LM probes (BERT-style)
  * Causal-LM generation probes (GPT-2-style)
  * Latent attractor reorientation (mid-layer hook; GPT-2)
  * Latent attractor telemetry (final hidden MOD BUS + similarity report)
  * Logging helpers for:
        - Embedding masked-LM bends
        - Embedding generation bends
        - Latent bends (final-layer MOD BUS)
        - Latent mid-layer bends
  * Config-driven runner: python nb_embedding_semantic_toolkit.py --config config.yml

Dependencies:
    torch
    transformers
    (optional) pyyaml for YAML configs
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

# =============================================================================
# BASIC MODEL / TOKENIZER LOADERS
# =============================================================================


def load_mlm(model_name: str = "bert-base-uncased", device: str | None = None):
    """
    Load a masked language model (MLM) and tokenizer.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer


def load_clm(model_name: str = "gpt2", device: str | None = None):
    """
    Load a causal language model (CLM) and tokenizer.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def load_gpt2_lm_head(model_name: str = "gpt2", device: str | None = None):
    """
    Load GPT-2 LMHead model + fast tokenizer for latent bends.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()
    return model, tokenizer


# =============================================================================
# EMBEDDING UTILITIES
# =============================================================================


def get_embedding_matrix(model: nn.Module) -> torch.Tensor:
    """
    Get the input embedding weight matrix from a transformer model.
    Returns [vocab_size, hidden_dim].
    """
    return model.get_input_embeddings().weight


def set_embedding_matrix(model: nn.Module, new_emb: torch.Tensor) -> None:
    """
    Replace the model's input embedding matrix in-place.
    """
    model.get_input_embeddings().weight.data = new_emb


def phrase_to_vector(
    phrase: str,
    tokenizer,
    embedding: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a word/phrase to a single vector by averaging embeddings
    of all non-special tokens.
    """
    encoded = tokenizer(
        phrase,
        add_special_tokens=False,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"][0]

    vecs = []
    for tid in token_ids.tolist():
        vecs.append(embedding[tid])

    if not vecs:
        raise ValueError(f"No usable tokens for phrase: {phrase!r}")

    return torch.stack(vecs, dim=0).mean(dim=0)


def compute_centroid(
    phrases: Sequence[str],
    tokenizer,
    embedding: torch.Tensor,
) -> torch.Tensor:
    """
    Compute centroid vector for a list of phrases.
    """
    vecs = []
    for p in phrases:
        try:
            v = phrase_to_vector(p, tokenizer, embedding)
            vecs.append(v)
        except ValueError:
            continue

    if not vecs:
        raise ValueError("No valid phrases for centroid computation.")

    return torch.stack(vecs, dim=0).mean(dim=0)


# =============================================================================
# EMBEDDING BENDS: CLUSTER DRIFT (DIRECTIONAL)
# =============================================================================


def apply_cluster_drift(
    embedding: torch.Tensor,
    tokenizer,
    from_cluster: Sequence[str],
    to_cluster: Sequence[str],
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Directional / cluster drift:

        from_cluster -> to_cluster

    For each phrase in from_cluster, move its vector toward centroid(to_cluster):

        v' = v + alpha * (centroid_to - centroid_from)
    """
    new_emb = embedding.clone()
    centroid_from = compute_centroid(from_cluster, tokenizer, embedding)
    centroid_to = compute_centroid(to_cluster, tokenizer, embedding)
    delta = centroid_to - centroid_from

    for phrase in from_cluster:
        encoded = tokenizer(
            phrase,
            add_special_tokens=False,
            return_tensors="pt",
        )
        for tid in encoded["input_ids"][0].tolist():
            new_emb[tid] = embedding[tid] + alpha * delta

    return new_emb


# =============================================================================
# EMBEDDING BENDS: PAIRWISE DRIFT
# =============================================================================


def apply_pairwise_drift(
    embedding: torch.Tensor,
    tokenizer,
    pairs: Sequence[Tuple[str, str]],
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Pairwise drift for concept pairs:

        src -> tgt

    v_src' = v_src + alpha * (v_tgt - v_src)
    """
    new_emb = embedding.clone()

    for src, tgt in pairs:
        try:
            v_src = phrase_to_vector(src, tokenizer, embedding)
            v_tgt = phrase_to_vector(tgt, tokenizer, embedding)
        except ValueError:
            continue

        encoded_src = tokenizer(
            src,
            add_special_tokens=False,
            return_tensors="pt",
        )
        delta = v_tgt - v_src
        for tid in encoded_src["input_ids"][0].tolist():
            new_emb[tid] = embedding[tid] + alpha * delta

    return new_emb


# =============================================================================
# EMBEDDING BENDS: SEMANTIC INVERSION (MIDPOINT REFLECTION)
# =============================================================================


def apply_semantic_inversion(
    embedding: torch.Tensor,
    tokenizer,
    pairs: Sequence[Tuple[str, str]],
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Semantic inversion for pairs (a, b).

    mid = 0.5 * (a + b)
    inv_a = a + alpha * ((2*mid - a) - a)
    inv_b = b + alpha * ((2*mid - b) - b)
    """
    new_emb = embedding.clone()

    for a_phrase, b_phrase in pairs:
        try:
            a_vec = phrase_to_vector(a_phrase, tokenizer, embedding)
            b_vec = phrase_to_vector(b_phrase, tokenizer, embedding)
        except ValueError:
            continue

        mid = 0.5 * (a_vec + b_vec)
        inv_a = a_vec + alpha * ((2 * mid - a_vec) - a_vec)
        inv_b = b_vec + alpha * ((2 * mid - b_vec) - b_vec)

        encoded_a = tokenizer(
            a_phrase,
            add_special_tokens=False,
            return_tensors="pt",
        )
        encoded_b = tokenizer(
            b_phrase,
            add_special_tokens=False,
            return_tensors="pt",
        )

        for tid in encoded_a["input_ids"][0].tolist():
            new_emb[tid] = inv_a
        for tid in encoded_b["input_ids"][0].tolist():
            new_emb[tid] = inv_b

    return new_emb


# =============================================================================
# MASKED-LM PROBES
# =============================================================================


def masked_top_k(
    model,
    tokenizer,
    prompt: str,
    k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Run a masked-LM probe and return top-k predictions at [MASK].
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    mask_id = tokenizer.mask_token_id

    mask_positions = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0]
    if mask_positions.numel() != 1:
        raise ValueError(f"Expected exactly one [MASK] in: {prompt}")

    mask_pos = mask_positions.item()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, mask_pos]

    scores, indices = torch.topk(logits, k)
    probs = torch.softmax(scores, dim=0).tolist()
    tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
    return list(zip(tokens, probs))


def run_masked_probe_suite(
    model,
    tokenizer,
    prompts: Sequence[str],
    role_vocab: Sequence[str] | None = None,
    k: int = 10,
) -> Dict[str, Dict[str, int]]:
    """
    Run a suite of masked-LM prompts and count how often specified
    role tokens appear in top-k predictions.
    """
    role_counts: Dict[str, int] = {r: 0 for r in (role_vocab or [])}

    for p in prompts:
        topk = masked_top_k(model, tokenizer, p, k=k)
        if role_vocab:
            for tok, _ in topk:
                if tok in role_counts:
                    role_counts[tok] += 1

    return {"role_counts": role_counts}


# =============================================================================
# CAUSAL-LM GENERATION PROBES
# =============================================================================


@dataclass
class GenerationConfig:
    max_new_tokens: int = 120
    num_samples: int = 8
    temperature: float = 0.9
    top_k: int = 50


def generate_many(
    model,
    tokenizer,
    prompt: str,
    cfg: GenerationConfig,
) -> List[str]:
    """
    Generate multiple samples from a causal LM with top-k sampling.
    """
    device = next(model.parameters()).device
    texts: List[str] = []

    for i in range(cfg.num_samples):
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        torch.manual_seed(42 + i)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                pad_token_id=tokenizer.pad_token_id,
            )

        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        texts.append(text)

    return texts


def count_token_occurrences(
    texts: Sequence[str],
    token_list: Sequence[str],
) -> Dict[str, int]:
    """
    Lowercase substring counting of tokens in generated texts.
    """
    counts: Dict[str, int] = {t: 0 for t in token_list}
    for t in texts:
        lower = t.lower()
        for tok in token_list:
            counts[tok] += lower.count(tok.lower())
    return counts


def generation_probe_suite(
    base_model,
    bent_model,
    tokenizer,
    prompts: Sequence[str],
    target_tokens: Sequence[str],
    cfg: GenerationConfig | None = None,
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Compare baseline vs bent causal LM across prompts with token-count telemetry.
    """
    if cfg is None:
        cfg = GenerationConfig()

    results: Dict[str, Dict[str, Dict[str, int]]] = {}

    for p in prompts:
        base_texts = generate_many(base_model, tokenizer, p, cfg)
        bent_texts = generate_many(bent_model, tokenizer, p, cfg)

        base_counts = count_token_occurrences(base_texts, target_tokens)
        bent_counts = count_token_occurrences(bent_texts, target_tokens)

        results[p] = {
            "baseline_counts": base_counts,
            "bent_counts": bent_counts,
        }

    return results


# =============================================================================
# LATENT ATTRACTOR REORIENTATION (MID-LAYER HOOK; GPT-2)
# =============================================================================


@dataclass
class AttractorVectors:
    directions: List[torch.Tensor]

    @property
    def mean_direction(self) -> torch.Tensor:
        stacked = torch.stack(self.directions, dim=0)
        return stacked.mean(dim=0)


def compute_layer_attractor_vectors(
    tokenizer: GPT2TokenizerFast,
    model: GPT2LMHeadModel,
    phrases: Sequence[str],
    layer_index: int,
) -> AttractorVectors:
    """
    Encode attractor phrases and collect average hidden states from a given layer.
    """
    vectors: List[torch.Tensor] = []
    device = next(model.parameters()).device

    for phrase in phrases:
        encoded = tokenizer(phrase, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_index].squeeze(0)
        vec = hidden.mean(dim=0)
        vectors.append(vec)

    return AttractorVectors(vectors)


def register_layer_attractor_hook(
    model: GPT2LMHeadModel,
    layer_index: int,
    bias_vector: torch.Tensor,
    lmbda: float,
):
    """
    Register a forward hook on a transformer block to add a scaled attractor vector
    to the hidden state at that layer.
    """
    bias = bias_vector * lmbda

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden_state = output[0] + bias
            return (hidden_state, *output[1:])
        else:
            return output + bias

    transformer = model.transformer
    block = transformer.h[layer_index]
    return block.register_forward_hook(hook)


def gpt2_generate_simple(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompts: Sequence[str],
    max_length: int = 40,
) -> List[str]:
    """
    Simple helper: generate continuations for a list of prompts with GPT-2.
    """
    outputs: List[str] = []
    device = next(model.parameters()).device

    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = encoded["input_ids"]
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                temperature=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        outputs.append(tokenizer.decode(generated[0], skip_special_tokens=True))

    return outputs


# =============================================================================
# LATENT ATTRACTOR TELEMETRY (FINAL HIDDEN MOD BUS; GPT-2)
# =============================================================================


@dataclass
class LatentBendReport:
    attractor_tokens: List[str]
    baseline_mean_similarity: float
    bent_mean_similarity: float
    baseline_similarities: List[float]
    bent_similarities: List[float]
    prompt_length: int


def build_attractor_vector_from_tokens(
    tokenizer,
    model: GPT2LMHeadModel,
    token_list: Sequence[str],
) -> torch.Tensor:
    """
    Build an attractor vector by averaging the embeddings of specified tokens.
    """
    embedding = model.transformer.wte.weight
    vecs = []
    for tok in token_list:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id == tokenizer.unk_token_id:
            continue
        vecs.append(embedding[tok_id])
    if not vecs:
        raise ValueError("No valid tokens for attractor.")
    return torch.stack(vecs, dim=0).mean(dim=0)


def decode_greedy(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompt: str,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, str]:
    """
    Greedy decoding without latent bend, returning token IDs and decoded text.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return outputs, tokenizer.decode(outputs[0], skip_special_tokens=True)


def decode_bent(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompt: str,
    max_new_tokens: int,
    attractor_vector: torch.Tensor,
    alpha: float,
    mix: float,
) -> Tuple[torch.Tensor, str]:
    """
    MOD BUS decoding: inject attractor vector into final hidden state and
    decode step-wise with greedy argmax.
    """
    device = next(model.parameters()).device
    attractor = attractor_vector.to(device).view(1, 1, -1)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            bent_hidden = hidden + alpha * attractor
            mixed_hidden = (1 - mix) * hidden + mix * bent_hidden
            logits = model.lm_head(mixed_hidden)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    return input_ids, tokenizer.decode(input_ids[0], skip_special_tokens=True)


def cosine_similarity_scores(
    embedding_matrix: torch.Tensor,
    ids: Sequence[int],
    attractor_vector: torch.Tensor,
) -> List[float]:
    """
    Cosine similarities between each token's embedding and the attractor vector.
    """
    if not ids:
        return []
    emb_rows = embedding_matrix[torch.tensor(ids, device=embedding_matrix.device)]
    attractor = attractor_vector.to(embedding_matrix.device)
    sims = nn.functional.cosine_similarity(emb_rows, attractor, dim=-1)
    return sims.detach().cpu().tolist()


def collect_similarity_report(
    embedding_matrix: torch.Tensor,
    prompt_len: int,
    baseline_ids: torch.Tensor,
    bent_ids: torch.Tensor,
    attractor_vector: torch.Tensor,
) -> LatentBendReport:
    """
    Build a similarity report for baseline vs bent sequences, ignoring prompt tokens.
    """
    baseline_seq = baseline_ids[0].tolist()
    bent_seq = bent_ids[0].tolist()

    baseline_new = baseline_seq[prompt_len:]
    bent_new = bent_seq[prompt_len:]

    baseline_sims = cosine_similarity_scores(embedding_matrix, baseline_new, attractor_vector)
    bent_sims = cosine_similarity_scores(embedding_matrix, bent_new, attractor_vector)

    baseline_mean = float(sum(baseline_sims) / max(len(baseline_sims), 1) if baseline_sims else 0.0)
    bent_mean = float(sum(bent_sims) / max(len(bent_sims), 1) if bent_sims else 0.0)

    return LatentBendReport(
        attractor_tokens=[],
        baseline_mean_similarity=baseline_mean,
        bent_mean_similarity=bent_mean,
        baseline_similarities=baseline_sims,
        bent_similarities=bent_sims,
        prompt_length=prompt_len,
    )


# =============================================================================
# LOGGING HELPERS FOR THEORY-USEFUL EXPERIMENTS
# =============================================================================


def diff_topk_lists(
    base_topk: List[Tuple[str, float]],
    bent_topk: List[Tuple[str, float]],
) -> Dict:
    """
    Compare two top-k lists ([(token, prob), ...]) and summarize changes.
    """
    k = min(len(base_topk), len(bent_topk))
    changes = []
    num_changed = 0

    for i in range(k):
        b_tok, b_prob = base_topk[i]
        bt_tok, bt_prob = bent_topk[i]
        changed = (b_tok != bt_tok)
        if changed:
            num_changed += 1

        changes.append(
            {
                "rank": i + 1,
                "base_token": b_tok,
                "base_prob": float(b_prob),
                "bent_token": bt_tok,
                "bent_prob": float(bt_prob),
                "changed": changed,
            }
        )

    return {
        "identical": num_changed == 0 and len(base_topk) == len(bent_topk),
        "num_positions_changed": num_changed,
        "changes": changes,
    }


def log_embedding_masked_bend_run(
    log_dir: str,
    bend_name: str,
    axis: str,
    mode: str,
    model_name: str,
    prompts: Sequence[str],
    base_topk_map: Dict[str, List[Tuple[str, float]]],
    bent_topk_map: Dict[str, List[Tuple[str, float]]],
    role_vocab: Sequence[str] | None = None,
    bend_params: Dict | None = None,
) -> str:
    """
    Log a masked-LM embedding/semantic bend experiment.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"embedding_bend_masked_{bend_name}_{timestamp}.json"
    path = os.path.join(log_dir, fname)

    role_vocab = list(role_vocab or [])
    base_roles = {r: 0 for r in role_vocab}
    bent_roles = {r: 0 for r in role_vocab}

    per_prompt = {}
    for p in prompts:
        base_topk = base_topk_map[p]
        bent_topk = bent_topk_map[p]
        diff = diff_topk_lists(base_topk, bent_topk)

        if role_vocab:
            for tok, _ in base_topk:
                if tok in base_roles:
                    base_roles[tok] += 1
            for tok, _ in bent_topk:
                if tok in bent_roles:
                    bent_roles[tok] += 1

        per_prompt[p] = {
            "base_topk": [
                {"token": t, "prob": float(prob)} for (t, prob) in base_topk
            ],
            "bent_topk": [
                {"token": t, "prob": float(prob)} for (t, prob) in bent_topk
            ],
            "diff": diff,
        }

    payload = {
        "timestamp": timestamp,
        "bend_name": bend_name,
        "axis": axis,
        "mode": mode,
        "model_name": model_name,
        "bend_params": bend_params or {},
        "prompts": list(prompts),
        "role_vocab": role_vocab,
        "role_counts": {
            "baseline": base_roles,
            "bent": bent_roles,
        },
        "per_prompt": per_prompt,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


def log_embedding_generation_bend_run(
    log_dir: str,
    bend_name: str,
    axis: str,
    mode: str,
    model_name: str,
    prompts: Sequence[str],
    target_tokens: Sequence[str],
    gen_results: Dict[str, Dict[str, Dict[str, int]]],
    bend_params: Dict | None = None,
) -> str:
    """
    Log a generation-based embedding/semantic bend experiment.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"embedding_bend_generation_{bend_name}_{timestamp}.json"
    path = os.path.join(log_dir, fname)

    agg_baseline = {t: 0 for t in target_tokens}
    agg_bent = {t: 0 for t in target_tokens}

    for p in prompts:
        base_counts = gen_results[p]["baseline_counts"]
        bent_counts = gen_results[p]["bent_counts"]
        for tok in target_tokens:
            agg_baseline[tok] += base_counts.get(tok, 0)
            agg_bent[tok] += bent_counts.get(tok, 0)

    payload = {
        "timestamp": timestamp,
        "bend_name": bend_name,
        "axis": axis,
        "mode": mode,
        "model_name": model_name,
        "bend_params": bend_params or {},
        "prompts": list(prompts),
        "target_tokens": list(target_tokens),
        "aggregate_counts": {
            "baseline": agg_baseline,
            "bent": agg_bent,
        },
        "per_prompt_counts": gen_results,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


def log_latent_bend_run(
    log_dir: str,
    bend_name: str,
    axis: str,
    mode: str,
    args: Dict,
    baseline_text: str,
    bent_text: str,
    report: LatentBendReport,
    token_list: Sequence[str],
) -> str:
    """
    Log latent-bend (MOD BUS) run to JSON file, with axis/mode metadata.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"latent_bend_{bend_name}_{timestamp}.json"
    path = os.path.join(log_dir, fname)

    payload = {
        "timestamp": timestamp,
        "bend_name": bend_name,
        "axis": axis,
        "mode": mode,
        "args": args,
        "baseline_text": baseline_text,
        "bent_text": bent_text,
        "report": asdict(report) | {"attractor_tokens": list(token_list)},
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


def log_latent_midlayer_run(
    log_dir: str,
    bend_name: str,
    axis: str,
    mode: str,
    model_name: str,
    args: Dict,
    prompts: Sequence[str],
    baseline_texts: Sequence[str],
    bent_texts: Sequence[str],
    attractor_phrases: Sequence[str],
    layer_index: int,
    lmbda: float,
) -> str:
    """
    Log a mid-layer latent attractor (Plate V) run.

    Stores:
      - prompts
      - baseline vs bent outputs
      - attractor phrases
      - layer / lambda
      - axis/mode/bend metadata
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"latent_midlayer_{bend_name}_{timestamp}.json"
    path = os.path.join(log_dir, fname)

    per_prompt = {}
    for p, b_txt, bent_txt in zip(prompts, baseline_texts, bent_texts):
        per_prompt[p] = {
            "baseline_text": b_txt,
            "bent_text": bent_txt,
        }

    payload = {
        "timestamp": timestamp,
        "bend_name": bend_name,
        "axis": axis,
        "mode": mode,
        "model_name": model_name,
        "args": args,
        "attractor_phrases": list(attractor_phrases),
        "layer_index": layer_index,
        "lambda": float(lmbda),
        "per_prompt": per_prompt,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


def print_similarity_meter(report: LatentBendReport) -> None:
    """
    Tiny similarity meter in text form.
    """
    print("=== Latent Attractor Similarity Meter ===")
    print(f"Baseline mean similarity: {report.baseline_mean_similarity:.4f}")
    print(f"Bent mean similarity:     {report.bent_mean_similarity:.4f}")
    delta = report.bent_mean_similarity - report.baseline_mean_similarity
    print(f"Delta:                    {delta:+.4f}")
    print("========================================")


# =============================================================================
# CONFIG LOADING + DISPATCH
# =============================================================================


def load_experiment_config(path: str) -> Dict:
    """
    Load an experiment config from YAML or JSON.

    - .yml / .yaml → parsed with PyYAML (requires `pip install pyyaml`)
    - otherwise   → parsed as JSON
    """
    with open(path, "r") as f:
        text = f.read()

    if path.endswith((".yml", ".yaml")):
        try:
            import yaml  # type: ignore
        except ImportError:
            raise RuntimeError(
                "PyYAML is required for YAML configs. Install with `pip install pyyaml` "
                "or use a JSON config instead."
            )
        return yaml.safe_load(text)
    else:
        return json.loads(text)


def run_experiment_from_config(cfg: Dict) -> str:
    """
    Dispatch experiments based on a config dict.

    Supported cfg keys (high level):

      experiment_type: "embedding_masked" | "embedding_generation" |
                       "latent_modbus" | "latent_midlayer"
      bend_name: str
      axis: str ("embedding/semantic", "latent", ...)
      mode: str ("revelatory", "recoherent", ...)
      log_dir: str

    For embedding_masked:

      model_name: str
      prompts: [str, ...]
      role_vocab: [str, ...]
      bend:
        type: "semantic_inversion" | "cluster_drift" | "pairwise_drift"
        alpha: float
        pairs: [[a, b], ...]        # for semantic_inversion or pairwise_drift
        from_cluster: [str, ...]    # for cluster_drift
        to_cluster: [str, ...]

    For embedding_generation:

      model_name: str
      prompts: [str, ...]
      target_tokens: [str, ...]
      gen_config:
        max_new_tokens: int
        num_samples: int
        temperature: float
        top_k: int
      bend: (same shape as above)

    For latent_modbus:

      model_name: str ("gpt2" recommended)
      prompt: str
      max_new_tokens: int
      attractor_tokens: [str, ...]
      alpha: float
      mix: float

    For latent_midlayer:

      model_name: str ("gpt2" recommended)
      prompts: [str, ...]
      attractor_phrases: [str, ...]
      layer_index: int
      lambda: float
      max_length: int
    """
    experiment_type = cfg["experiment_type"]
    bend_name = cfg.get("bend_name", experiment_type)
    axis = cfg.get("axis", "embedding/semantic")
    mode = cfg.get("mode", "revelatory")
    log_dir = cfg.get("log_dir", "runs")

    if experiment_type == "embedding_masked":
        model_name = cfg.get("model_name", "bert-base-uncased")
        prompts = cfg["prompts"]
        role_vocab = cfg.get("role_vocab", [])
        bend_cfg = cfg["bend"]

        mlm, tok = load_mlm(model_name)
        emb = get_embedding_matrix(mlm)

        bend_type = bend_cfg["type"]
        alpha = float(bend_cfg.get("alpha", 0.5))

        if bend_type == "semantic_inversion":
            pairs = bend_cfg["pairs"]
            new_emb = apply_semantic_inversion(emb, tok, pairs, alpha=alpha)
        elif bend_type == "cluster_drift":
            from_cluster = bend_cfg["from_cluster"]
            to_cluster = bend_cfg["to_cluster"]
            new_emb = apply_cluster_drift(emb, tok, from_cluster, to_cluster, alpha=alpha)
            pairs = []  # for logging
        elif bend_type == "pairwise_drift":
            pairs = bend_cfg["pairs"]
            new_emb = apply_pairwise_drift(emb, tok, pairs, alpha=alpha)
        else:
            raise ValueError(f"Unknown bend.type: {bend_type}")

        bent_mlm = AutoModelForMaskedLM.from_pretrained(model_name)
        bent_mlm.to(next(mlm.parameters()).device)
        set_embedding_matrix(bent_mlm, new_emb)
        bent_mlm.eval()

        base_topk_map = {p: masked_top_k(mlm, tok, p, k=10) for p in prompts}
        bent_topk_map = {p: masked_top_k(bent_mlm, tok, p, k=10) for p in prompts}

        bend_params = {
            "bend_type": bend_type,
            "alpha": alpha,
            "pairs": bend_cfg.get("pairs", []),
            "from_cluster": bend_cfg.get("from_cluster", []),
            "to_cluster": bend_cfg.get("to_cluster", []),
        }

        log_path = log_embedding_masked_bend_run(
            log_dir=os.path.join(log_dir, "embedding_masked"),
            bend_name=bend_name,
            axis=axis,
            mode=mode,
            model_name=model_name,
            prompts=prompts,
            base_topk_map=base_topk_map,
            bent_topk_map=bent_topk_map,
            role_vocab=role_vocab,
            bend_params=bend_params,
        )
        return log_path

    elif experiment_type == "embedding_generation":
        model_name = cfg.get("model_name", "gpt2")
        prompts = cfg["prompts"]
        target_tokens = cfg["target_tokens"]
        bend_cfg = cfg["bend"]
        gen_cfg_dict = cfg.get("gen_config", {})
        gen_cfg = GenerationConfig(
            max_new_tokens=int(gen_cfg_dict.get("max_new_tokens", 120)),
            num_samples=int(gen_cfg_dict.get("num_samples", 8)),
            temperature=float(gen_cfg_dict.get("temperature", 0.9)),
            top_k=int(gen_cfg_dict.get("top_k", 50)),
        )

        clm, clm_tok = load_clm(model_name)
        emb = get_embedding_matrix(clm)

        bend_type = bend_cfg["type"]
        alpha = float(bend_cfg.get("alpha", 0.5))

        if bend_type == "semantic_inversion":
            pairs = bend_cfg["pairs"]
            new_emb = apply_semantic_inversion(emb, clm_tok, pairs, alpha=alpha)
        elif bend_type == "cluster_drift":
            from_cluster = bend_cfg["from_cluster"]
            to_cluster = bend_cfg["to_cluster"]
            new_emb = apply_cluster_drift(emb, clm_tok, from_cluster, to_cluster, alpha=alpha)
            pairs = []
        elif bend_type == "pairwise_drift":
            pairs = bend_cfg["pairs"]
            new_emb = apply_pairwise_drift(emb, clm_tok, pairs, alpha=alpha)
        else:
            raise ValueError(f"Unknown bend.type: {bend_type}")

        bent_clm = AutoModelForCausalLM.from_pretrained(model_name)
        bent_clm.to(next(clm.parameters()).device)
        set_embedding_matrix(bent_clm, new_emb)
        bent_clm.config.pad_token_id = clm_tok.pad_token_id
        bent_clm.eval()

        gen_results = generation_probe_suite(
            clm,
            bent_clm,
            clm_tok,
            prompts=prompts,
            target_tokens=target_tokens,
            cfg=gen_cfg,
        )

        bend_params = {
            "bend_type": bend_type,
            "alpha": alpha,
            "pairs": bend_cfg.get("pairs", []),
            "from_cluster": bend_cfg.get("from_cluster", []),
            "to_cluster": bend_cfg.get("to_cluster", []),
            "gen_config": asdict(gen_cfg),
        }

        log_path = log_embedding_generation_bend_run(
            log_dir=os.path.join(log_dir, "embedding_generation"),
            bend_name=bend_name,
            axis=axis,
            mode=mode,
            model_name=model_name,
            prompts=prompts,
            target_tokens=target_tokens,
            gen_results=gen_results,
            bend_params=bend_params,
        )
        return log_path

    elif experiment_type == "latent_modbus":
        model_name = cfg.get("model_name", "gpt2")
        prompt = cfg["prompt"]
        max_new_tokens = int(cfg.get("max_new_tokens", 40))
        attractor_tokens = cfg["attractor_tokens"]
        alpha = float(cfg.get("alpha", 0.5))
        mix = float(cfg.get("mix", 1.0))

        gpt2_model, gpt2_tok = load_gpt2_lm_head(model_name)
        attractor_vec = build_attractor_vector_from_tokens(
            gpt2_tok, gpt2_model, attractor_tokens
        )

        baseline_ids, baseline_text = decode_greedy(
            gpt2_model, gpt2_tok, prompt, max_new_tokens
        )
        bent_ids, bent_text = decode_bent(
            gpt2_model, gpt2_tok, prompt, max_new_tokens, attractor_vec, alpha=alpha, mix=mix
        )

        embedding_matrix = gpt2_model.transformer.wte.weight
        prompt_len = len(gpt2_tok.encode(prompt))
        report = collect_similarity_report(
            embedding_matrix, prompt_len, baseline_ids, bent_ids, attractor_vec
        )

        args = {
            "prompt": prompt,
            "alpha": alpha,
            "mix": mix,
            "max_new_tokens": max_new_tokens,
            "model_name": model_name,
        }

        log_path = log_latent_bend_run(
            log_dir=os.path.join(log_dir, "latent_modbus"),
            bend_name=bend_name,
            axis=axis,
            mode=mode,
            args=args,
            baseline_text=baseline_text,
            bent_text=bent_text,
            report=report,
            token_list=attractor_tokens,
        )

        print_similarity_meter(report)
        return log_path

    elif experiment_type == "latent_midlayer":
        model_name = cfg.get("model_name", "gpt2")
        prompts = cfg["prompts"]
        attractor_phrases = cfg["attractor_phrases"]
        layer_index = int(cfg.get("layer_index", 6))
        lmbda = float(cfg.get("lambda", 0.5))
        max_length = int(cfg.get("max_length", 60))

        gpt2_model, gpt2_tok = load_gpt2_lm_head(model_name)

        # 1) Compute attractor vectors at the specified layer
        av = compute_layer_attractor_vectors(
            tokenizer=gpt2_tok,
            model=gpt2_model,
            phrases=attractor_phrases,
            layer_index=layer_index,
        )
        bias_vec = av.mean_direction

        # 2) Baseline texts
        baseline_texts = gpt2_generate_simple(
            model=gpt2_model,
            tokenizer=gpt2_tok,
            prompts=prompts,
            max_length=max_length,
        )

        # 3) Register hook + bent texts
        hook = register_layer_attractor_hook(
            model=gpt2_model,
            layer_index=layer_index,
            bias_vector=bias_vec,
            lmbda=lmbda,
        )
        try:
            bent_texts = gpt2_generate_simple(
                model=gpt2_model,
                tokenizer=gpt2_tok,
                prompts=prompts,
                max_length=max_length,
            )
        finally:
            hook.remove()

        args = {
            "prompts": prompts,
            "attractor_phrases": attractor_phrases,
            "layer_index": layer_index,
            "lambda": lmbda,
            "max_length": max_length,
            "model_name": model_name,
        }

        log_path = log_latent_midlayer_run(
            log_dir=os.path.join(log_dir, "latent_midlayer"),
            bend_name=bend_name,
            axis=axis,
            mode=mode,
            model_name=model_name,
            args=args,
            prompts=prompts,
            baseline_texts=baseline_texts,
            bent_texts=bent_texts,
            attractor_phrases=attractor_phrases,
            layer_index=layer_index,
            lmbda=lmbda,
        )
        return log_path

    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")


# =============================================================================
# MAIN: CONFIG-DRIVEN ENTRYPOINT
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Neural Bending Toolkit — Embedding/Semantic + Latent Bends (config-driven)."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to experiment config file (.yml/.yaml or .json).",
        required=True,
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    log_path = run_experiment_from_config(cfg)
    print(f"\n[nbt] Experiment complete. Log written to:\n  {log_path}\n")
