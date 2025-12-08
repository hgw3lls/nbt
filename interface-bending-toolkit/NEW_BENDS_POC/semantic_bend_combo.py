# semantic_bend_combo.py

"""
Compare four conditions on the same masked prompts:

1) baseline BERT
2) directional drift only (neighborhood-style)
3) inversion only (semantic reflection)
4) drift + inversion combined

Assumes:
- `semantic_inversion.py` defines apply_semantic_inversion + top_k_mask_predictions
- `embedding_neighborhood_drift.py` defines compute_centroid + build_drifted_embedding_matrix

Neither of those should import this file.
"""

import copy
import json
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from semantic_inversion import apply_semantic_inversion, top_k_mask_predictions
import embedding_neighborhood_drift as neighborhood


MODEL_NAME = "bert-base-uncased"
TOP_K = 10
DRIFT_ALPHA = 0.6

PROMPTS = [
    "The role of the [MASK] is to protect the neighborhood.",
    "When there is a crisis, the [MASK] is called.",
]

# Inversion pairs (semantic mirror)
INVERSION_PAIRS = [
    ("police", "community care"),
    ("citizen", "resident"),
]

# Directional drift cluster:
DOMINANT_WORDS = ["police"]
MARGINALIZED_WORDS = ["community care"]


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return model, tokenizer


def make_directionally_drifted_model(base_model, tokenizer):
    """Apply neighborhood-style directional drift to MARGINALIZED_WORDS -> DOMINANT_WORDS."""
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


def make_inverted_model(base_model, tokenizer):
    """Apply semantic inversion bend only."""
    model = copy.deepcopy(base_model)
    base_emb = model.get_input_embeddings().weight.data
    inv_emb = apply_semantic_inversion(
        base_emb,
        tokenizer,
        INVERSION_PAIRS,
        alpha=0.6,
    )
    model.get_input_embeddings().weight.data = inv_emb
    return model


def make_combo_model(base_model, tokenizer):
    """Apply directional drift first, then inversion on top."""
    model = make_directionally_drifted_model(base_model, tokenizer)
    emb = model.get_input_embeddings().weight.data
    combo_emb = apply_semantic_inversion(
        emb,
        tokenizer,
        INVERSION_PAIRS,
        alpha=0.6,
    )
    model.get_input_embeddings().weight.data = combo_emb
    return model


def run_all():
    base_model, tokenizer = load_model_and_tokenizer()

    model_drift = make_directionally_drifted_model(base_model, tokenizer)
    model_invert = make_inverted_model(base_model, tokenizer)
    model_combo = make_combo_model(base_model, tokenizer)

    results: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}

    for prompt in PROMPTS:
        results[prompt] = {}
        results[prompt]["baseline"] = top_k_mask_predictions(
            base_model, tokenizer, prompt, k=TOP_K
        )
        results[prompt]["drift"] = top_k_mask_predictions(
            model_drift, tokenizer, prompt, k=TOP_K
        )
        results[prompt]["invert"] = top_k_mask_predictions(
            model_invert, tokenizer, prompt, k=TOP_K
        )
        results[prompt]["combo"] = top_k_mask_predictions(
            model_combo, tokenizer, prompt, k=TOP_K
        )

    # Pretty-print table
    for prompt in PROMPTS:
        print("=" * 100)
        print(f"PROMPT: {prompt}")
        print("Rank | Baseline                | Drift                   | Invert                  | Combo")
        print("-" * 100)
        for i in range(TOP_K):
            b_tok, b_prob = results[prompt]["baseline"][i]
            d_tok, d_prob = results[prompt]["drift"][i]
            iv_tok, iv_prob = results[prompt]["invert"][i]
            c_tok, c_prob = results[prompt]["combo"][i]
            print(
                f"{i+1:>4} | "
                f"{b_tok:<22} ({b_prob:0.3f}) | "
                f"{d_tok:<23} ({d_prob:0.3f}) | "
                f"{iv_tok:<23} ({iv_prob:0.3f}) | "
                f"{c_tok:<23} ({c_prob:0.3f})"
            )
        print()

    # Log JSON as well
    json_out = {
        "model_name": MODEL_NAME,
        "prompts": PROMPTS,
        "inversion_pairs": INVERSION_PAIRS,
        "dominant_words": DOMINANT_WORDS,
        "marginalized_words": MARGINALIZED_WORDS,
        "drift_alpha": DRIFT_ALPHA,
        "top_k": TOP_K,
        "results": {},
    }
    for p in PROMPTS:
        json_out["results"][p] = {}
        for cond in ["baseline", "drift", "invert", "combo"]:
            json_out["results"][p][cond] = [
                {"token": t, "prob": float(pr)}
                for (t, pr) in results[p][cond]
            ]

    with open("semantic_bend_combo_results.json", "w") as f:
        json.dump(json_out, f, indent=2)

    print("Logged to semantic_bend_combo_results.json")


if __name__ == "__main__":
    run_all()
