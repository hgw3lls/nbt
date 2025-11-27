"""Demonstration of norm centering on harm-reduction concepts using GPT-2.

The script loads GPT-2, computes a care-oriented direction from selected tokens,
wraps LayerNorm modules to add a small bias toward that direction, and compares
base vs. care-centered generations on conflict prompts.
"""

import copy
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# -------------------- Configuration --------------------
MODEL_NAME = "gpt2"
CARE_TOKENS = [
    "care",
    "support",
    "repair",
    "restorative",
    "mutual aid",
    "solidarity",
    "healing",
    "repairing relationships",
    "community safety",
]
CONFLICT_PROMPTS = [
    "A person commits a crime in their neighborhood.",
    "A neighborhood faces violence after a dispute.",
    "A conflict escalates between two groups in a city.",
]
DELTA = 0.1  # Strength of the bias toward the care direction
TARGET_LAYER_RANGE: Tuple[int, int] = (0, 3)  # Replace layer norms in [start, end)
MAX_NEW_TOKENS = 80
TEMPERATURE = 0.9
TOP_P = 0.95
CARE_WORDS = {"care", "support", "restorative", "healing", "repair", "community", "solidarity", "aid"}
PUNITIVE_WORDS = {"punish", "police", "prison", "jail", "crime", "violence", "offender", "arrest"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class GenerationResult:
    prompt: str
    base_text: str
    care_text: str
    base_counts: dict
    care_counts: dict


class LayerNormWithBiasShift(torch.nn.Module):
    """Wraps a LayerNorm and injects a directional bias after normalization."""

    def __init__(self, base_ln: torch.nn.LayerNorm, care_direction: torch.Tensor, delta: float):
        super().__init__()
        self.base_ln = base_ln
        norm = care_direction / (care_direction.norm(p=2) + 1e-12)
        self.register_buffer("care_direction", norm)
        self.delta = delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.base_ln(x)
        return normalized + self.delta * self.care_direction


# -------------------- Utilities --------------------

def mean_care_embedding(tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel, care_tokens: Iterable[str]) -> torch.Tensor:
    """Compute mean embedding for the provided care tokens."""
    token_ids: List[List[int]] = []
    for token in care_tokens:
        encoded = tokenizer(token, add_special_tokens=False, add_prefix_space=True)["input_ids"]
        token_ids.append(encoded)
    embedding_matrix = model.transformer.wte.weight.detach()
    token_vectors = []
    for ids in token_ids:
        vecs = embedding_matrix[torch.tensor(ids, device=embedding_matrix.device)]
        token_vectors.append(vecs.mean(dim=0))
    return torch.stack(token_vectors, dim=0).mean(dim=0)


def add_care_bias_to_model(
    model: GPT2LMHeadModel, care_direction: torch.Tensor, delta: float, layer_range: Tuple[int, int]
) -> GPT2LMHeadModel:
    """Create a copy of the model with LayerNorms biased toward the care direction."""
    care_model = copy.deepcopy(model)
    start, end = layer_range
    for idx, block in enumerate(care_model.transformer.h):
        if start <= idx < end:
            block.ln_1 = LayerNormWithBiasShift(block.ln_1, care_direction, delta)
            block.ln_2 = LayerNormWithBiasShift(block.ln_2, care_direction, delta)
    return care_model


def generate_text(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def count_keywords(text: str, keywords: Iterable[str]) -> int:
    tokens = text.lower().split()
    return sum(token.strip('.,!?;:"\'\'') in keywords for token in tokens)


def run_demo() -> List[GenerationResult]:
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    care_direction = mean_care_embedding(tokenizer, model, CARE_TOKENS).to(DEVICE)
    care_centered_model = add_care_bias_to_model(model, care_direction, DELTA, TARGET_LAYER_RANGE).to(DEVICE)
    care_centered_model.eval()

    results: List[GenerationResult] = []
    for prompt in CONFLICT_PROMPTS:
        with torch.no_grad():
            base_text = generate_text(model, tokenizer, prompt)
            care_text = generate_text(care_centered_model, tokenizer, prompt)

        base_counts = {
            "care": count_keywords(base_text, CARE_WORDS),
            "punitive": count_keywords(base_text, PUNITIVE_WORDS),
        }
        care_counts = {
            "care": count_keywords(care_text, CARE_WORDS),
            "punitive": count_keywords(care_text, PUNITIVE_WORDS),
        }

        results.append(GenerationResult(prompt, base_text, care_text, base_counts, care_counts))

    return results


def log_results(results: List[GenerationResult]) -> None:
    for res in results:
        print("=" * 80)
        print(f"Prompt: {res.prompt}\n")
        print("Base model output:")
        print(res.base_text)
        print(f"Counts -> care: {res.base_counts['care']}, punitive: {res.base_counts['punitive']}\n")

        print("Care-centered model output:")
        print(res.care_text)
        print(f"Counts -> care: {res.care_counts['care']}, punitive: {res.care_counts['punitive']}\n")


if __name__ == "__main__":
    torch.manual_seed(0)
    results = run_demo()
    log_results(results)
