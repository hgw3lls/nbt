import argparse
import json
import os
from copy import deepcopy
from datetime import datetime
from typing import List, Optional, Tuple

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessor


# INVERSION PATCH: swapping stabilization profiles across the stack to bend where context is aggregated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LayerNorm inversion patch for GPT-style models.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt to condition on")
    parser.add_argument("--invert_pairs", type=str, default="0:11,1:10", help="Comma-separated layer index pairs to swap, e.g., '0:11,1:10'")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Number of tokens to generate")
    parser.add_argument("--mix", type=float, default=1.0, help="Blend between baseline and inverted logits")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model identifier (gpt2 or distilgpt2)")
    return parser.parse_args()


def parse_pairs(pairs_str: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    if not pairs_str:
        return pairs
    for chunk in pairs_str.split(","):
        if ":" not in chunk:
            continue
        left, right = chunk.split(":", maxsplit=1)
        pairs.append((int(left.strip()), int(right.strip())))
    return pairs


def load_model_and_tokenizer(model_name: str):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


class MixedLogitsProcessor(LogitsProcessor):
    """Blend logits from baseline and inverted runs to simulate partial inversion."""

    def __init__(self, baseline_model: GPT2LMHeadModel, mix: float):
        super().__init__()
        self.baseline_model = baseline_model
        self.mix = mix

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        with torch.no_grad():
            baseline_logits = self.baseline_model(input_ids).logits[:, -1, :]
        # CROSS-WIRE: combining inverted gain stage with original reference
        return (1 - self.mix) * baseline_logits + self.mix * scores


def swap_layernorms(model: GPT2LMHeadModel, pairs: List[Tuple[int, int]]):
    diffs = []
    num_layers = len(model.transformer.h)
    for a_idx, b_idx in pairs:
        if a_idx < 0 or b_idx < 0 or a_idx >= num_layers or b_idx >= num_layers:
            raise ValueError(f"Layer pair ({a_idx}, {b_idx}) is out of bounds for model with {num_layers} blocks")
        block_a = model.transformer.h[a_idx]
        block_b = model.transformer.h[b_idx]

        for ln_name in ["ln_1", "ln_2"]:
            if not hasattr(block_a, ln_name) or not hasattr(block_b, ln_name):
                continue
            ln_a = getattr(block_a, ln_name)
            ln_b = getattr(block_b, ln_name)
            with torch.no_grad():
                weight_diff = torch.norm(ln_a.weight - ln_b.weight).item()
                bias_diff = torch.norm(ln_a.bias - ln_b.bias).item()
                # SWAP: flipping gain stages between early and late stabilizers
                temp_w = ln_a.weight.clone()
                temp_b = ln_a.bias.clone()
                ln_a.weight.copy_(ln_b.weight)
                ln_a.bias.copy_(ln_b.bias)
                ln_b.weight.copy_(temp_w)
                ln_b.bias.copy_(temp_b)
            diffs.append({
                "pair": f"{a_idx}:{b_idx}",
                "layernorm": ln_name,
                "weight_l2_before_swap": weight_diff,
                "bias_l2_before_swap": bias_diff,
            })
    return diffs


def generate_text(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int,
    logits_processor: Optional[List[LogitsProcessor]] = None,
) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
    }
    if logits_processor:
        generation_kwargs["logits_processor"] = logits_processor
    with torch.no_grad():
        output_ids = model.generate(input_ids, **generation_kwargs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def log_run(payload: dict, folder: str = "runs/layernorm_inversion") -> None:
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"layernorm_inversion_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Logged run to {path}")


def main():
    args = parse_args()
    pairs = parse_pairs(args.invert_pairs)
    if not pairs:
        raise ValueError("No invert pairs provided; use the format '0:11,1:10'")

    baseline_model, tokenizer = load_model_and_tokenizer(args.model_name)
    baseline_model.eval()

    print("Generating baseline (no inversion)...")
    baseline_text = generate_text(baseline_model, tokenizer, args.prompt, args.max_new_tokens)

    print("Cloning model for inversion...")
    inverted_model = deepcopy(baseline_model)

    print("Applying LayerNorm swaps to inverted model...")
    ln_diffs = swap_layernorms(inverted_model, pairs)

    for diff in ln_diffs:
        print(
            f"Swapped {diff['layernorm']} for pair {diff['pair']} | weight Δ: {diff['weight_l2_before_swap']:.4f}, bias Δ: {diff['bias_l2_before_swap']:.4f}"
        )

    processor = [MixedLogitsProcessor(baseline_model, args.mix)] if args.mix < 1.0 else None

    print("Generating with inverted LayerNorms...")
    inverted_text = generate_text(
        inverted_model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        logits_processor=processor,
    )

    payload = {
        "prompt": args.prompt,
        "invert_pairs": args.invert_pairs,
        "max_new_tokens": args.max_new_tokens,
        "mix": args.mix,
        "model_name": args.model_name,
        "baseline_text": baseline_text,
        "inverted_text": inverted_text,
        "layernorm_diffs": ln_diffs,
    }
    log_run(payload)

    print("\n=== Output Comparison ===")
    print("-- Baseline --")
    print(baseline_text)
    print("\n-- Inverted --")
    print(inverted_text)

    print("\nNote: Swapping stabilization profiles between early and late layers bends the model's internal sense of where context is aggregated.")


if __name__ == "__main__":
    main()
