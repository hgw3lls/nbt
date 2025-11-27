"""
Circuit-bent latent attractor reorientation.

This script injects an attractor vector into the latent bus of a small
causal LM (GPT-2) to see how decoding drifts toward a conceptual cluster.
Baseline and bent generations are compared along with similarity telemetry.
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SimilarityReport:
    baseline_mean: float
    bent_mean: float
    baseline_values: List[float]
    bent_values: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent attractor reorientation")
    parser.add_argument("--prompt", required=True, help="Prompt used for generation")
    parser.add_argument(
        "--attractor_tokens",
        default="care,repair,mutual,community",
        help="Comma-separated tokens defining attractor direction",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Strength of attractor")
    parser.add_argument("--max_new_tokens", type=int, default=60, help="Tokens to generate")
    parser.add_argument(
        "--mix",
        type=float,
        default=1.0,
        help="Dry/wet mix between baseline and attractor-bent hidden state [0,1]",
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


def embed_token_list(tokenizer: AutoTokenizer, tokens: Sequence[str]) -> List[int]:
    ids: List[int] = []
    for tok in tokens:
        encoded = tokenizer.encode(tok, add_special_tokens=False)
        if not encoded:
            raise ValueError(f"Token '{tok}' produced no ids with tokenizer {tokenizer.name_or_path}")
        ids.extend(encoded)
    return ids


def compute_attractor_vector(embedding_matrix: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    rows = embedding_matrix[torch.tensor(ids, device=embedding_matrix.device)]
    vector = rows.mean(dim=0)
    norm = torch.norm(vector) + 1e-8
    return vector / norm


def decode_greedy(model, tokenizer, prompt: str, max_new_tokens: int) -> Tuple[torch.Tensor, str]:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    return input_ids, tokenizer.decode(input_ids[0], skip_special_tokens=True)


def decode_bent(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    attractor_vector: torch.Tensor,
    alpha: float,
    mix: float,
) -> Tuple[torch.Tensor, str]:
    device = next(model.parameters()).device
    attractor = attractor_vector.to(device).view(1, 1, -1)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            # MOD BUS: patching an attractor vector into the latent bus to see where the model's world settles.
            bent_hidden = hidden + alpha * attractor
            mixed_hidden = (1 - mix) * hidden + mix * bent_hidden
            logits = model.lm_head(mixed_hidden)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

    return input_ids, tokenizer.decode(input_ids[0], skip_special_tokens=True)


def cosine_similarity_scores(
    embedding_matrix: torch.Tensor, ids: Sequence[int], attractor_vector: torch.Tensor
) -> List[float]:
    if not ids:
        return []
    emb_rows = embedding_matrix[torch.tensor(ids, device=embedding_matrix.device)]
    attractor = attractor_vector.to(embedding_matrix.device)
    sims = nn.functional.cosine_similarity(emb_rows, attractor, dim=-1)
    return sims.detach().cpu().tolist()


def average_similarity(scores: Sequence[float]) -> float:
    if not scores:
        return float("nan")
    return float(sum(scores) / len(scores))


def collect_similarity_report(
    embedding_matrix: torch.Tensor,
    prompt_len: int,
    baseline_ids: torch.Tensor,
    bent_ids: torch.Tensor,
    attractor_vector: torch.Tensor,
) -> SimilarityReport:
    base_new = baseline_ids[0, prompt_len:].tolist()
    bent_new = bent_ids[0, prompt_len:].tolist()
    baseline_scores = cosine_similarity_scores(embedding_matrix, base_new, attractor_vector)
    bent_scores = cosine_similarity_scores(embedding_matrix, bent_new, attractor_vector)
    return SimilarityReport(
        baseline_mean=average_similarity(baseline_scores),
        bent_mean=average_similarity(bent_scores),
        baseline_values=baseline_scores,
        bent_values=bent_scores,
    )


def log_run(
    log_dir: str,
    args: argparse.Namespace,
    baseline_text: str,
    bent_text: str,
    report: SimilarityReport,
    token_list: Sequence[str],
) -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"latent_bend_{timestamp}.json")

    payload = {
        "prompt": args.prompt,
        "attractor_tokens": list(token_list),
        "alpha": args.alpha,
        "mix": args.mix,
        "max_new_tokens": args.max_new_tokens,
        "baseline": baseline_text,
        "bent": bent_text,
        "baseline_similarity_mean": report.baseline_mean,
        "bent_similarity_mean": report.bent_mean,
        "baseline_similarity_values": report.baseline_values,
        "bent_similarity_values": report.bent_values,
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return log_path


def print_similarity_meter(report: SimilarityReport):
    def bar(value: float) -> str:
        if value != value:  # NaN check
            return "|" + "?" * 10 + "|"
        length = 10
        clamped = max(-1.0, min(1.0, value))
        filled = int((clamped + 1) / 2 * length)
        return "|" + "#" * filled + "-" * (length - filled) + "|"

    print("Similarity drift meter (baseline vs bent):")
    print(f"  baseline mean: {report.baseline_mean:.4f} {bar(report.baseline_mean)}")
    print(f"  bent mean    : {report.bent_mean:.4f} {bar(report.bent_mean)}")


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    attractor_tokens = parse_token_list(args.attractor_tokens)
    embedding_matrix = model.get_input_embeddings().weight.detach()
    attractor_ids = embed_token_list(tokenizer, attractor_tokens)
    attractor_vector = compute_attractor_vector(embedding_matrix, attractor_ids)

    prompt_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    prompt_len = prompt_ids.shape[1]

    baseline_ids, baseline_text = decode_greedy(model, tokenizer, args.prompt, args.max_new_tokens)
    bent_ids, bent_text = decode_bent(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        attractor_vector,
        args.alpha,
        args.mix,
    )

    report = collect_similarity_report(
        embedding_matrix, prompt_len, baseline_ids, bent_ids, attractor_vector
    )

    log_path = log_run(
        log_dir="runs/latent_bend",
        args=args,
        baseline_text=baseline_text,
        bent_text=bent_text,
        report=report,
        token_list=attractor_tokens,
    )

    print("# MOD BUS: attractor vector patched; routing log to", log_path)
    print_similarity_meter(report)
    print("Baseline output:\n", baseline_text)
    print("Bent output:\n", bent_text)


if __name__ == "__main__":
    main()
