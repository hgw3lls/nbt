"""
semantic_bend_experiment.py

Standalone experiment that does NOT import from semantic_inversion.py.

It implements:
- semantic inversion (reflection across concept midpoints)
- neighborhood-style directional drift (using embedding_neighborhood_drift)
- masked LM probes for "police / community care" style roles

Conditions compared:
1) baseline BERT
2) inversion only
3) directional drift only
4) directional drift + inversion combined
"""

import copy
import json
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

import embedding_neighborhood_drift as neighborhood


MODEL_NAME = "bert-base-uncased"
TOP_K = 10
DRIFT_ALPHA = 0.6
INV_ALPHA = 0.6

PROMPTS = [
    "The role of the [MASK] is to protect the neighborhood.",
    "When there is a crisis, the [MASK] is called.",
]

# Semantic inversion pairs
INVERSION_PAIRS = [
    ("police", "community care"),
    ("citizen", "resident"),
]

# Directional drift cluster
DOMINANT_WORDS = ["police"]
MARGINALIZED_WORDS = ["community care"]


# ---------------------------------------------------------------------
# Core helpers: load model, semantic inversion, top-k predictions
# ---------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return model, tokenizer


def semantic_inversion(
    embedding_matrix: torch.Tensor,
    tokenizer,
    concept_pairs,
    alpha: float = 0.6,
) -> torch.Tensor:
    """
    Reflect selected concept embeddings across their midpoint.

    For each pair (a, b):
        mid = 0.5 * (a + b)
        inv_a = a + alpha * ((2*mid - a) - a)
        inv_b = b + alpha * ((2*mid - b) - b)

    This pulls a and b toward each other's side of the midpoint.
    """
    new_embedding = embedding_matrix.clone()

    for a_tok, b_tok in concept_pairs:
        try:
            a_id = tokenizer.convert_tokens_to_ids(a_tok)
            b_id = tokenizer.convert_tokens_to_ids(b_tok)
        except Exception:
            continue

        if a_id == tokenizer.unk_token_id or b_id == tokenizer.unk_token_id:
            continue

        a_vec = new_embedding[a_id]
        b_vec = new_embedding[b_id]
        mid = 0.5 * (a_vec + b_vec)

        # reflect partially
        inv_a = a_vec + alpha * ((2 * mid - a_vec) - a_vec)
        inv_b = b_vec + alpha * ((2 * mid - b_vec) - b_vec)

        new_embedding[a_id] = inv_a
        new_embedding[b_id] = inv_b

    return new_embedding


def top_k_mask_predictions(
    model,
    tokenizer,
    prompt: str,
    k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Get top-k predictions at the [MASK] position.
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    mask_token_id = tokenizer.mask_token_id

    mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
    if mask_positions.numel() != 1:
        raise ValueError(f"Prompt must contain exactly one [MASK]: {prompt}")

    mask_index = mask_positions.item()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, mask_index]

    scores, indices = torch.topk(logits, k)
    tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
    # Convert to softmax probabilities for interpretability
    probs = torch.softmax(scores, dim=0).tolist()

    return list(zip(tokens, probs))


# ---------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------

def make_inverted_model(base_model, tokenizer):
    """
    Apply semantic inversion only.
    """
    model = copy.deepcopy(base_model)
    base_emb = model.get_input_embeddings().weight.data
    inv_emb = semantic_inversion(
        base_emb,
        tokenizer,
        INVERSION_PAIRS,
        alpha=INV_ALPHA,
    )
    model.get_input_embeddings().weight.data = inv_emb
    return model


def make_directionally_drifted_model(base_model, tokenizer):
    """
    Apply neighborhood-style directional drift to MARGINALIZED_WORDS -> DOMINANT_WORDS.
    """
    model = copy.deepcopy(base_model)
    emb = model.get_input_embeddings().weight.data

    dom_centroid = neighborhood.compute_centroid(DOMINANT_WORDS, tokenizer, emb)
    drifted_emb, _ = neighborhood.build_drifted_embedding_matrix(
        emb,
        MARGINALIZED_WORDS,
        tokenizer,
        dom_centroid,
        DRIFT_ALPHA,
    )
    model.get_input_embeddings().weight.data = drifted_emb
    return model


def make_combo_model(base_model, tokenizer):
    """
    Apply directional drift first, then semantic inversion on top.
    """
    model = make_directionally_drifted_model(base_model, tokenizer)
    emb = model.get_input_embeddings().weight.data
    combo_emb = semantic_inversion(
        emb,
        tokenizer,
        INVERSION_PAIRS,
        alpha=INV_ALPHA,
    )
    model.get_input_embeddings().weight.data = combo_emb
    return model


# ---------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------

def run():
    base_model, tokenizer = load_model_and_tokenizer()

    model_invert = make_inverted_model(base_model, tokenizer)
    model_drift = make_directionally_drifted_model(base_model, tokenizer)
    model_combo = make_combo_model(base_model, tokenizer)

    results: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}

    for prompt in PROMPTS:
        results[prompt] = {}
        results[prompt]["baseline"] = top_k_mask_predictions(
            base_model, tokenizer, prompt, k=TOP_K
        )
        results[prompt]["invert"] = top_k_mask_predictions(
            model_invert, tokenizer, prompt, k=TOP_K
        )
        results[prompt]["drift"] = top_k_mask_predictions(
            model_drift, tokenizer, prompt, k=TOP_K
        )
        results[prompt]["combo"] = top_k_mask_predictions(
            model_combo, tokenizer, prompt, k=TOP_K
        )

    # Pretty-print
    for prompt in PROMPTS:
        print("=" * 100)
        print(f"PROMPT: {prompt}")
        print("Rank | Baseline                | Invert                  | Drift                   | Combo")
        print("-" * 100)
        for i in range(TOP_K):
            b_tok, b_prob = results[prompt]["baseline"][i]
            iv_tok, iv_prob = results[prompt]["invert"][i]
            d_tok, d_prob = results[prompt]["drift"][i]
            c_tok, c_prob = results[prompt]["combo"][i]
            print(
                f"{i+1:>4} | "
                f"{b_tok:<22} ({b_prob:0.3f}) | "
                f"{iv_tok:<23} ({iv_prob:0.3f}) | "
                f"{d_tok:<23} ({d_prob:0.3f}) | "
                f"{c_tok:<23} ({c_prob:0.3f})"
            )
        print()

    # JSON log for later interpretation
    json_out = {
        "model_name": MODEL_NAME,
        "prompts": PROMPTS,
        "inversion_pairs": INVERSION_PAIRS,
        "dominant_words": DOMINANT_WORDS,
        "marginalized_words": MARGINALIZED_WORDS,
        "drift_alpha": DRIFT_ALPHA,
        "inv_alpha": INV_ALPHA,
        "top_k": TOP_K,
        "results": {},
    }
    for p in PROMPTS:
        json_out["results"][p] = {}
        for cond in ["baseline", "invert", "drift", "combo"]:
            json_out["results"][p][cond] = [
                {"token": t, "prob": float(pr)}
                for (t, pr) in results[p][cond]
            ]

    with open("semantic_bend_experiment_results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    print("Logged to semantic_bend_experiment_results.json")


if __name__ == "__main__":
    run()

