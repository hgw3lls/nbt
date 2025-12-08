# semantic_inversion_run.py

"""
Run the semantic inversion bend on its own and log results to JSON.

Assumes `semantic_inversion.py` defines:

    apply_semantic_inversion(embedding, tokenizer, concept_pairs, alpha)
    top_k_mask_predictions(model, tokenizer, prompt, k)

but DOES NOT import this file (to avoid circular imports).
"""

import copy
import json
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from semantic_inversion import apply_semantic_inversion, top_k_mask_predictions


MODEL_NAME = "bert-base-uncased"
ALPHA = 0.6
TOP_K = 10

PROMPTS = [
    "The role of the [MASK] is to protect the neighborhood.",
    "When there is a crisis, the [MASK] is called.",
]

CONCEPT_PAIRS = [
    ("police", "community care"),
    ("citizen", "resident"),
]


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return model, tokenizer


def main():
    model_orig, tokenizer = load_model_and_tokenizer()
    model_inv = copy.deepcopy(model_orig)

    base_emb = model_orig.get_input_embeddings().weight.data
    inv_emb = apply_semantic_inversion(
        base_emb,
        tokenizer,
        CONCEPT_PAIRS,
        alpha=ALPHA,
    )
    model_inv.get_input_embeddings().weight.data = inv_emb

    original_predictions: Dict[str, List[Tuple[str, float]]] = {}
    inverted_predictions: Dict[str, List[Tuple[str, float]]] = {}

    for prompt in PROMPTS:
        original_predictions[prompt] = top_k_mask_predictions(
            model_orig, tokenizer, prompt, k=TOP_K
        )
        inverted_predictions[prompt] = top_k_mask_predictions(
            model_inv, tokenizer, prompt, k=TOP_K
        )

    # Pretty-print to console
    for prompt in PROMPTS:
        print("=" * 80)
        print("PROMPT:", prompt)
        print(f"{'Rank':>4} | {'Original':<20} | {'Inverted':<20}")
        print("-" * 80)
        for i in range(TOP_K):
            o_tok, o_prob = original_predictions[prompt][i]
            iv_tok, iv_prob = inverted_predictions[prompt][i]
            print(
                f"{i+1:>4} | "
                f"{o_tok:<20} ({o_prob:.3f}) | "
                f"{iv_tok:<20} ({iv_prob:.3f})"
            )
        print()

    # Log to JSON
    out = {
        "model_name": MODEL_NAME,
        "alpha": ALPHA,
        "top_k": TOP_K,
        "concept_pairs": CONCEPT_PAIRS,
        "results": [],
    }
    for p in PROMPTS:
        out["results"].append(
            {
                "prompt": p,
                "original": [
                    {"token": t, "prob": float(pr)} for (t, pr) in original_predictions[p]
                ],
                "inverted": [
                    {"token": t, "prob": float(pr)} for (t, pr) in inverted_predictions[p]
                ],
            }
        )

    with open("semantic_inversion_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Logged to semantic_inversion_results.json")


if __name__ == "__main__":
    main()
