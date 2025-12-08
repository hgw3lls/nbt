"""
semantic_inversion_run.py

Run the semantic inversion bend on its own and log results to JSON
so you can diff multiple runs or inspect in your tools.
"""

import copy
import json
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from semantic_inversion import (
    load_model_and_tokenizer,
    apply_semantic_inversion,
    top_k_mask_predictions,
    print_comparisons,
)


def main():
    prompts = [
        "The role of the [MASK] is to protect the neighborhood.",
        "When there is a crisis, the [MASK] is called.",
    ]

    concept_pairs = [
        ("police", "community care"),
        ("citizen", "resident"),
    ]

    model_original, tokenizer = load_model_and_tokenizer()

    model_inverted = copy.deepcopy(model_original)
    base_embedding = model_original.get_input_embeddings().weight.data
    inverted_embedding = apply_semantic_inversion(
        base_embedding, tokenizer, concept_pairs, alpha=0.6
    )
    model_inverted.get_input_embeddings().weight.data = inverted_embedding

    k = 10
    original_predictions: Dict[str, List[Tuple[str, float]]] = {}
    inverted_predictions: Dict[str, List[Tuple[str, float]]] = {}

    for prompt in prompts:
        original_predictions[prompt] = top_k_mask_predictions(
            model_original, tokenizer, prompt, k=k
        )
        inverted_predictions[prompt] = top_k_mask_predictions(
            model_inverted, tokenizer, prompt, k=k
        )

    # Pretty-print to console as before
    print_comparisons(original_predictions, inverted_predictions, k)

    # Also log to JSON for later analysis
    output = {
        "concept_pairs": concept_pairs,
        "alpha": 0.6,
        "k": k,
        "results": [],
    }
    for p in prompts:
        output["results"].append(
            {
                "prompt": p,
                "original": [
                    {"token": t, "prob": float(prob)}
                    for (t, prob) in original_predictions[p]
                ],
                "inverted": [
                    {"token": t, "prob": float(prob)}
                    for (t, prob) in inverted_predictions[p]
                ],
            }
        )

    with open("semantic_inversion_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nLogged to semantic_inversion_results.json")


if __name__ == "__main__":
    main()
