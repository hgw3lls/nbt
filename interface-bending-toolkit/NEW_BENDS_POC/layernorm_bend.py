"""
Circuit-bent layer norm surrogate: bias logits toward a care-centered reference.

This script simulates bending the model's sense of "normal" without retraining by
applying logit biases driven by similarities to a care/harm-reduction vocabulary.
Baseline and bent generations are compared and logged with similarity telemetry.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Circuit-bend layer norm reference")
    parser.add_argument("--prompt", required=True, help="Prompt to seed generation")
    parser.add_argument("--mode", choices=["variance", "care_center"], required=True)
    parser.add_argument(
        "--care_tokens",
        default="care,repair,collective,community",
        help="Comma-separated tokens defining care/harm-reduction vocabulary",
    )
    parser.add_argument("--variance_scale", type=float, default=1.0, help="Noise scale")
    parser.add_argument("--care_scale", type=float, default=5.0, help="Care bias scale")
    parser.add_argument("--max_new_tokens", type=int, default=60, help="Tokens to generate")
    parser.add_argument("--mix", type=float, default=1.0, help="Dry/wet mix for bent logits")
    parser.add_argument("--model_name", default="gpt2", help="Hugging Face model name")
    return parser.parse_args()


def parse_token_list(raw: str) -> List[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def care_centroid(tokenizer, tokens: Iterable[str], embedding_matrix: torch.Tensor) -> torch.Tensor:
    ids: List[int] = []
    for tok in tokens:
        encoded = tokenizer.encode(tok, add_special_tokens=False)
        if not encoded:
            raise ValueError(f"Token '{tok}' produced no ids for tokenizer {tokenizer.name_or_path}")
        ids.extend(encoded)
    rows = embedding_matrix[torch.tensor(ids, device=embedding_matrix.device)]
    centroid = rows.mean(dim=0)
    norm = torch.norm(centroid) + 1e-8
    return centroid / norm


def similarity_vectors(embedding_matrix: torch.Tensor, centroid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    normed_embed = embedding_matrix / (embedding_matrix.norm(dim=1, keepdim=True) + 1e-8)
    dot_sim = torch.matmul(normed_embed, centroid)
    distance = torch.norm(normed_embed - centroid, dim=1)
    return dot_sim, distance


def ascii_histogram(values: Sequence[float], bins: int = 10) -> str:
    if not values:
        return "(no tokens generated)"
    min_v, max_v = min(values), max(values)
    span = max(max_v - min_v, 1e-6)
    counts = [0 for _ in range(bins)]
    for v in values:
        idx = int((v - min_v) / span * (bins - 1))
        counts[idx] += 1
    max_count = max(counts)
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * (50 * c // max_count if max_count else 0)
        low = min_v + (span / bins) * i
        high = min_v + (span / bins) * (i + 1)
        lines.append(f"[{low:+.2f}, {high:+.2f}]: {bar}")
    return "\n".join(lines)


def bend_logits(
    logits: torch.Tensor,
    mode: str,
    care_scale: float,
    variance_scale: float,
    similarity: torch.Tensor,
    distance: torch.Tensor,
    mix: float,
) -> torch.Tensor:
    # REGULATOR BEND: shifting the model's voltage reference of 'normal' toward care-centered tokens.
    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be between 0 and 1")
    bent = logits
    if mode == "care_center":
        bent = logits + care_scale * similarity
    elif mode == "variance":
        noise = torch.randn_like(logits) * variance_scale * (1 + distance)
        bent = logits + noise
    return (1 - mix) * logits + mix * bent


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    bent: bool,
    mode: str,
    care_scale: float,
    variance_scale: float,
    mix: float,
    similarity: torch.Tensor,
    distance: torch.Tensor,
) -> Tuple[str, List[float]]:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    sim_list: List[float] = []
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            step_logits = outputs.logits[:, -1, :]
            if bent:
                adjusted = bend_logits(step_logits, mode, care_scale, variance_scale, similarity, distance, mix)
            else:
                adjusted = step_logits
            next_token = torch.argmax(adjusted, dim=-1, keepdim=True)
            sim_list.append(similarity[next_token].item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return decoded, sim_list


def log_run(
    log_dir: str,
    args: argparse.Namespace,
    baseline_text: str,
    bent_text: str,
    similarity_scores: List[float],
) -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"layernorm_bend_{timestamp}.json")

    payload: Dict[str, object] = {
        "prompt": args.prompt,
        "mode": args.mode,
        "care_tokens": parse_token_list(args.care_tokens),
        "variance_scale": args.variance_scale,
        "care_scale": args.care_scale,
        "mix": args.mix,
        "max_new_tokens": args.max_new_tokens,
        "baseline": baseline_text,
        "bent": bent_text,
        "similarity_scores": similarity_scores,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return log_path


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    care_list = parse_token_list(args.care_tokens)
    embedding_matrix = model.get_input_embeddings().weight.detach().to(device)
    centroid = care_centroid(tokenizer, care_list, embedding_matrix)
    similarity, distance = similarity_vectors(embedding_matrix, centroid)

    baseline_text, baseline_scores = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        bent=False,
        mode=args.mode,
        care_scale=args.care_scale,
        variance_scale=args.variance_scale,
        mix=args.mix,
        similarity=similarity,
        distance=distance,
    )

    bent_text, bent_scores = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        bent=True,
        mode=args.mode,
        care_scale=args.care_scale,
        variance_scale=args.variance_scale,
        mix=args.mix,
        similarity=similarity,
        distance=distance,
    )

    log_dir = os.path.join("runs", "layernorm_bend")
    log_path = log_run(log_dir, args, baseline_text, bent_text, bent_scores)

    print("=== LAYERNORM BEND REPORT ===")
    print(f"Prompt: {args.prompt}")
    print(f"Mode: {args.mode} | Mix: {args.mix} | Care scale: {args.care_scale} | Variance scale: {args.variance_scale}")
    print(f"Care tokens: {care_list}")
    print(f"Log saved to: {log_path}")
    print("-- Baseline --")
    print(baseline_text)
    print("-- Bent --")
    print(bent_text)
    print("-- Care-similarity histogram (bent run) --")
    print(ascii_histogram(bent_scores))


if __name__ == "__main__":
    main()
