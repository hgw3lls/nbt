"""
Substrate drift toward structural analysis.

This script nudges GPT-2 decoding toward systemic / relational language and
away from individualizing or moralizing frames by bending logits in-flight.
Baseline and bent generations are compared along with token count telemetry.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Iterable, List, Sequence, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor


class StructuralDriftProcessor(LogitsProcessor):
    """Logits processor that globally biases systemic vs individual terms."""

    def __init__(
        self,
        structural_ids: Sequence[int],
        individual_ids: Sequence[int],
        structural_scale: float,
        individual_scale: float,
        mix: float,
    ) -> None:
        super().__init__()
        self.structural_ids = torch.tensor(structural_ids, dtype=torch.long)
        self.individual_ids = torch.tensor(individual_ids, dtype=torch.long)
        self.structural_scale = structural_scale
        self.individual_scale = individual_scale
        self.mix = mix

    def to(self, device: torch.device) -> "StructuralDriftProcessor":
        self.structural_ids = self.structural_ids.to(device)
        self.individual_ids = self.individual_ids.to(device)
        return self

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # GLOBAL DRIFT: nudging the decoding substrate toward systemic language.
        bent_scores = scores.clone()
        if self.structural_ids.numel() > 0:
            bent_scores[..., self.structural_ids] += self.structural_scale
        if self.individual_ids.numel() > 0:
            bent_scores[..., self.individual_ids] += self.individual_scale
        return (1 - self.mix) * scores + self.mix * bent_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Substrate drift toward structural analysis")
    parser.add_argument("--prompt", required=True, help="Prompt to seed generation")
    parser.add_argument(
        "--structural_tokens",
        default="history,system,structure,infrastructure,policy,collective",
        help="Comma-separated systemic vocabulary",
    )
    parser.add_argument(
        "--individual_tokens",
        default="blame,criminal,bad,personal,individual",
        help="Comma-separated individualizing vocabulary",
    )
    parser.add_argument("--structural_scale", type=float, default=4.0, help="Boost for structural tokens")
    parser.add_argument(
        "--individual_scale",
        type=float,
        default=-4.0,
        help="Negative bias for individualizing tokens",
    )
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Tokens to generate")
    parser.add_argument(
        "--mix",
        type=float,
        default=1.0,
        help="Dry/wet mix between baseline and bent logits [0,1]",
    )
    parser.add_argument("--model_name", default="gpt2", help="Hugging Face model to use")
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def parse_tokens(raw: str) -> List[str]:
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def tokens_to_ids(tokenizer, tokens: Iterable[str]) -> List[int]:
    ids: List[int] = []
    for tok in tokens:
        encoded = tokenizer.encode(tok, add_special_tokens=False)
        if not encoded:
            continue
        ids.extend(encoded)
    return ids


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> Tuple[torch.Tensor, str]:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return output, tokenizer.decode(output[0], skip_special_tokens=True)


def generate_bent(
    model,
    tokenizer,
    prompt: str,
    processor: StructuralDriftProcessor,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, str]:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model_kwargs = {"logits_processor": [processor.to(device)]}
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            **model_kwargs,
        )
    return output, tokenizer.decode(output[0], skip_special_tokens=True)


def count_token_hits(tokenizer, text: str, vocab_ids: Set[int]) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return sum(1 for i in ids if i in vocab_ids)


def log_run(
    log_dir: str,
    args: argparse.Namespace,
    structural_tokens: Sequence[str],
    individual_tokens: Sequence[str],
    baseline_text: str,
    bent_text: str,
    structural_hits: int,
    individual_hits: int,
) -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"structural_drift_{timestamp}.json")
    payload = {
        "prompt": args.prompt,
        "structural_tokens": list(structural_tokens),
        "individual_tokens": list(individual_tokens),
        "structural_scale": args.structural_scale,
        "individual_scale": args.individual_scale,
        "mix": args.mix,
        "max_new_tokens": args.max_new_tokens,
        "baseline": baseline_text,
        "bent": bent_text,
        "structural_hits": structural_hits,
        "individual_hits": individual_hits,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def print_counts(structural_hits: int, individual_hits: int) -> None:
    total = structural_hits + individual_hits
    ratios = {
        "structural": structural_hits,
        "individual": individual_hits,
    }
    print("Token tendency counts (bent output):")
    for label, count in ratios.items():
        bar_len = 20
        filled = 0 if total == 0 else int(bar_len * count / max(total, 1))
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"  {label:10s}: {count:3d} |{bar}|")


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    structural_tokens = parse_tokens(args.structural_tokens)
    individual_tokens = parse_tokens(args.individual_tokens)

    structural_ids = tokens_to_ids(tokenizer, structural_tokens)
    individual_ids = tokens_to_ids(tokenizer, individual_tokens)

    baseline_ids, baseline_text = generate_text(
        model, tokenizer, args.prompt, args.max_new_tokens
    )

    processor = StructuralDriftProcessor(
        structural_ids=structural_ids,
        individual_ids=individual_ids,
        structural_scale=args.structural_scale,
        individual_scale=args.individual_scale,
        mix=args.mix,
    )

    bent_ids, bent_text = generate_bent(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
    )

    structural_hit_count = count_token_hits(
        tokenizer, bent_text, vocab_ids=set(structural_ids)
    )
    individual_hit_count = count_token_hits(
        tokenizer, bent_text, vocab_ids=set(individual_ids)
    )

    log_path = log_run(
        log_dir=os.path.join("runs", "structural_drift"),
        args=args,
        structural_tokens=structural_tokens,
        individual_tokens=individual_tokens,
        baseline_text=baseline_text,
        bent_text=bent_text,
        structural_hits=structural_hit_count,
        individual_hits=individual_hit_count,
    )

    print("== Structural Drift Patch ==")
    print(f"Prompt: {args.prompt}")
    print(f"Structural tokens: {structural_tokens}")
    print(f"Individual tokens: {individual_tokens}")
    print(f"Baseline output:\n{baseline_text}\n")
    print(f"Bent output:\n{bent_text}\n")
    print_counts(structural_hit_count, individual_hit_count)
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()
