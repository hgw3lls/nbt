"""
Circuit-bent residual stream demo.

This script wraps a small causal LM (GPT-2 by default) with a residual stream
post-processor that simulates bottlenecking or relational nudging before logits
are produced. Baseline and bent generations are logged alongside hidden-state
metrics to showcase how bending affects internal signals without permanently
altering model weights.
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BendMetrics:
    cosine_similarity: float
    baseline_singular: List[float]
    bent_singular: List[float]
    baseline_last_norm: float
    bent_last_norm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Circuit-bend residual stream")
    parser.add_argument("--prompt", required=True, help="Prompt used for generation")
    parser.add_argument("--mode", choices=["bottleneck", "relational"], required=True)
    parser.add_argument("--bottleneck_dim", type=int, default=64, help="Dim for bottleneck")
    parser.add_argument(
        "--relational_tokens",
        default="community,repair,mutual,system",
        help="Comma-separated tokens defining relational direction",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Bend strength [0,1]")
    parser.add_argument("--max_new_tokens", type=int, default=60, help="Tokens to generate")
    parser.add_argument(
        "--mix",
        type=float,
        default=1.0,
        help="Dry/wet mix between baseline and bent hidden states [0,1]",
    )
    parser.add_argument("--model_name", default="gpt2", help="Hugging Face model name")
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def parse_token_list(raw: str) -> List[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def pooled_embedding(embedding_matrix: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    rows = embedding_matrix[torch.tensor(ids, device=embedding_matrix.device)]
    return rows.mean(dim=0)


def relational_direction(tokenizer: AutoTokenizer, tokens: Iterable[str], embedding_matrix: torch.Tensor) -> torch.Tensor:
    ids: List[int] = []
    for tok in tokens:
        encoded = tokenizer.encode(tok, add_special_tokens=False)
        if not encoded:
            raise ValueError(f"Token '{tok}' produced no ids with tokenizer {tokenizer.name_or_path}")
        ids.extend(encoded)
    vector = pooled_embedding(embedding_matrix, ids)
    norm = torch.norm(vector) + 1e-8
    return vector / norm


def bottleneck(hidden_states: torch.Tensor, bottleneck_dim: int, alpha: float) -> torch.Tensor:
    hidden_dim = hidden_states.size(-1)
    dim = max(1, min(bottleneck_dim, hidden_dim))
    # CHOKE POINT: bottlenecking residual stream to minimal grammar
    projector = torch.eye(hidden_dim, device=hidden_states.device)[:dim]
    compressed = torch.matmul(hidden_states, projector.t())
    restored = torch.matmul(compressed, projector)
    return hidden_states + alpha * (restored - hidden_states)


def relational_bend(hidden_states: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    # FEEDBACK PATCH: nudging narrative spine toward relational direction vector
    return hidden_states + alpha * direction


def bend_hidden_states(
    hidden_states: torch.Tensor,
    mode: str,
    alpha: float,
    bottleneck_dim: int,
    direction: torch.Tensor,
) -> torch.Tensor:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if mode == "bottleneck":
        return bottleneck(hidden_states, bottleneck_dim, alpha)
    return relational_bend(hidden_states, direction, alpha)


def cosine_similarity_tensor(a: torch.Tensor, b: torch.Tensor) -> float:
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    return nn.functional.cosine_similarity(flat_a, flat_b, dim=0).item()


def hidden_metrics(baseline: torch.Tensor, bent: torch.Tensor) -> BendMetrics:
    baseline_s = torch.linalg.svdvals(baseline[0]).detach().cpu()
    bent_s = torch.linalg.svdvals(bent[0]).detach().cpu()
    return BendMetrics(
        cosine_similarity=cosine_similarity_tensor(baseline, bent),
        baseline_singular=baseline_s[:5].tolist(),
        bent_singular=bent_s[:5].tolist(),
        baseline_last_norm=torch.norm(baseline[0, -1]).item(),
        bent_last_norm=torch.norm(bent[0, -1]).item(),
    )


def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def generate_bent(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    mode: str,
    alpha: float,
    mix: float,
    bottleneck_dim: int,
    direction: torch.Tensor,
) -> str:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            bent_hidden = bend_hidden_states(hidden, mode, alpha, bottleneck_dim, direction)
            mixed_hidden = (1 - mix) * hidden + mix * bent_hidden
            logits = model.lm_head(mixed_hidden)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    # Recompute bent hidden on final sequence for logging
    final_outputs = model(input_ids, output_hidden_states=True)
    final_hidden = final_outputs.hidden_states[-1]
    final_bent_hidden = bend_hidden_states(final_hidden, mode, alpha, bottleneck_dim, direction)
    _ = (1 - mix) * final_hidden + mix * final_bent_hidden

    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return decoded


def log_run(
    log_dir: str,
    args: argparse.Namespace,
    baseline_text: str,
    bent_text: str,
    metrics: BendMetrics,
    token_list: Sequence[str],
) -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"residual_bend_{timestamp}.json")

    payload: Dict[str, object] = {
        "prompt": args.prompt,
        "mode": args.mode,
        "alpha": args.alpha,
        "mix": args.mix,
        "bottleneck_dim": args.bottleneck_dim,
        "relational_tokens": list(token_list),
        "baseline": baseline_text,
        "bent": bent_text,
        "cosine_similarity": metrics.cosine_similarity,
        "baseline_singular": metrics.baseline_singular,
        "bent_singular": metrics.bent_singular,
        "baseline_last_norm": metrics.baseline_last_norm,
        "bent_last_norm": metrics.bent_last_norm,
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return log_path


def print_gain_map(metrics: BendMetrics) -> None:
    print("\n=== RESIDUAL BEND TELEMETRY ===")
    print(f"Cosine similarity (baseline vs bent hidden): {metrics.cosine_similarity:.4f}")
    print("Singular values (top-5):")
    print(f"  baseline: {[round(v, 4) for v in metrics.baseline_singular]}")
    print(f"  bent    : {[round(v, 4) for v in metrics.bent_singular]}")
    print(
        f"Last token norms -> baseline: {metrics.baseline_last_norm:.4f}, bent: {metrics.bent_last_norm:.4f}"
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.mix <= 1.0:
        raise ValueError("mix must be between 0 and 1")
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    relational_tokens = parse_token_list(args.relational_tokens)
    embedding_matrix = model.get_input_embeddings().weight.detach()
    direction = relational_direction(tokenizer, relational_tokens, embedding_matrix)

    baseline_text = generate_baseline(model, tokenizer, args.prompt, args.max_new_tokens)
    bent_text = generate_bent(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.mode,
        args.alpha,
        args.mix,
        args.bottleneck_dim,
        direction,
    )

    # Compute metrics on prompt-only hidden states to avoid generation side-effects
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        base_outputs = model(prompt_ids, output_hidden_states=True)
        base_hidden = base_outputs.hidden_states[-1]
        bent_prompt_hidden = bend_hidden_states(
            base_hidden, args.mode, args.alpha, args.bottleneck_dim, direction
        )
        mixed_prompt_hidden = (1 - args.mix) * base_hidden + args.mix * bent_prompt_hidden

    metrics = hidden_metrics(base_hidden, mixed_prompt_hidden)
    print_gain_map(metrics)
    log_path = log_run(
        log_dir=os.path.join("runs", "residual_bend"),
        args=args,
        baseline_text=baseline_text,
        bent_text=bent_text,
        metrics=metrics,
        token_list=relational_tokens,
    )

    print(f"\n# PATCH REPORT written to {log_path}")


if __name__ == "__main__":
    main()
