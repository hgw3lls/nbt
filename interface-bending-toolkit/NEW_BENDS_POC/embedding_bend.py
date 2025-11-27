"""
Circuit-bent embedding demo.

This script perturbs selected token embeddings of a small causal LM (GPT-2 by
default) to simulate embedding drift or semantic inversion without permanently
modifying model weights. It logs before/after similarities, baseline output, and
bent output while providing a playful "oscilloscope" readout of drift strength.
"""

import argparse
import copy
import json
import os
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Circuit-bend transformer embeddings")
    parser.add_argument("--model_name", default="gpt2", help="Hugging Face model name")
    parser.add_argument("--prompt", required=True, help="Prompt used for generation")
    parser.add_argument(
        "--bend_mode",
        choices=["drift", "invert"],
        required=True,
        help="Type of embedding bend to apply",
    )
    parser.add_argument(
        "--source_tokens",
        required=True,
        help="Comma-separated list of source tokens to bend",
    )
    parser.add_argument(
        "--target_tokens",
        required=True,
        help="Comma-separated list of target tokens used for drift/inversion",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Bend strength [0,1]")
    parser.add_argument(
        "--mix",
        type=float,
        default=1.0,
        help="Dry/wet mix between original and bent logits [0,1]",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=60,
        help="Number of tokens to generate",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def token_ids(tokenizer: AutoTokenizer, text: str) -> List[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Token '{text}' produced no ids with tokenizer {tokenizer.name_or_path}")
    return ids


def pooled_embedding(embedding_matrix: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    rows = embedding_matrix[torch.tensor(ids, device=embedding_matrix.device)]
    return rows.mean(dim=0)


def apply_bend(
    embedding_matrix: torch.Tensor,
    tokenizer: AutoTokenizer,
    source_tokens: Iterable[str],
    target_tokens: Iterable[str],
    mode: str,
    alpha: float,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    modified = embedding_matrix.clone()
    metrics: List[Dict[str, float]] = []

    for src, tgt in zip(source_tokens, target_tokens):
        src_ids = token_ids(tokenizer, src)
        tgt_ids = token_ids(tokenizer, tgt)

        src_vec = pooled_embedding(modified, src_ids)
        tgt_vec = pooled_embedding(modified, tgt_ids)

        center = 0.5 * (src_vec + tgt_vec)

        if mode == "drift":
            delta = alpha * (tgt_vec - src_vec)
        else:
            reflected = 2 * center - src_vec
            delta = alpha * (reflected - src_vec)

        bent_vec = src_vec + delta

        # PATCH POINT: cross-wire f"{src}" with f"{tgt}" in embedding space
        for idx in src_ids:
            modified[idx] = bent_vec

        before_cos = nn.functional.cosine_similarity(src_vec, tgt_vec, dim=0).item()
        after_cos = nn.functional.cosine_similarity(bent_vec, tgt_vec, dim=0).item()
        drift_mag = torch.norm(delta).item()

        metrics.append(
            {
                "source": src,
                "target": tgt,
                "before_cos": before_cos,
                "after_cos": after_cos,
                "drift_magnitude": drift_mag,
            }
        )

    return modified, metrics


def generate_simple(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            next_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def generate_mixed(
    base_model,
    bent_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    mix: float,
) -> str:
    device = next(base_model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    base_model.eval()
    bent_model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            base_out = base_model(input_ids)
            bent_out = bent_model(input_ids)
            combined_logits = (1 - mix) * base_out.logits[:, -1, :] + mix * bent_out.logits[:, -1, :]
            next_token = torch.argmax(combined_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def oscilloscope(metrics: List[Dict[str, float]]) -> None:
    if not metrics:
        return
    max_mag = max(m["drift_magnitude"] for m in metrics) or 1e-8
    print("\n=== DRIFT OSCILLOSCOPE ===")
    for m in metrics:
        normalized = m["drift_magnitude"] / max_mag
        bar_len = max(1, int(normalized * 32))
        bar = "#" * bar_len
        print(f"{m['source']:<15} -> {m['target']:<15} |{bar}| {m['drift_magnitude']:.4f}")
    print()


def log_run(
    log_dir: str,
    args: argparse.Namespace,
    baseline: str,
    bent: str,
    metrics: List[Dict[str, float]],
    source_tokens: Sequence[str],
    target_tokens: Sequence[str],
) -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"embedding_bend_{timestamp}.json")

    payload = {
        "prompt": args.prompt,
        "bend_mode": args.bend_mode,
        "source_tokens": list(source_tokens),
        "target_tokens": list(target_tokens),
        "alpha": args.alpha,
        "mix": args.mix,
        "baseline": baseline,
        "bent": bent,
        "metrics": metrics,
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return log_path


def main():
    args = parse_args()
    source_tokens = [t.strip() for t in args.source_tokens.split(",") if t.strip()]
    target_tokens = [t.strip() for t in args.target_tokens.split(",") if t.strip()]

    if len(source_tokens) != len(target_tokens):
        raise ValueError("source_tokens and target_tokens must have the same length")
    if not source_tokens:
        raise ValueError("No tokens provided for bending")
    if not 0.0 <= args.mix <= 1.0:
        raise ValueError("mix should be between 0 and 1 for dry/wet control")

    model_base, tokenizer = load_model_and_tokenizer(args.model_name)
    model_bent = copy.deepcopy(model_base)

    base_embedding = model_base.get_input_embeddings().weight.detach()
    bent_embedding, metrics = apply_bend(
        base_embedding,
        tokenizer,
        source_tokens,
        target_tokens,
        args.bend_mode,
        args.alpha,
    )
    with torch.no_grad():
        model_bent.get_input_embeddings().weight.copy_(bent_embedding)

    baseline_text = generate_simple(
        model_base,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
    )

    bent_text = generate_mixed(
        model_base,
        model_bent,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        mix=args.mix,
    )

    log_path = log_run(
        "runs/embedding_bend",
        args,
        baseline_text,
        bent_text,
        metrics,
        source_tokens,
        target_tokens,
    )

    print(f"\nPrompt: {args.prompt}\n")
    print(f"Baseline output:\n{baseline_text}\n")
    print(f"Bent output ({args.bend_mode}, mix={args.mix}):\n{bent_text}\n")

    oscilloscope(metrics)
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
