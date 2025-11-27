"""
Synthetic perplexity injection: bend the sampler to favor unlikely tokens.

This script simulates a lying internal perplexity meter by boosting
low-probability tokens during decoding, revealing the model's latent tastes
when improbables are treated as optimal.
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor


@dataclass
class StepTrace:
    token_id: int
    token_text: str
    rank_under_p: int
    p_value: float


class SyntheticPerplexityProcessor(LogitsProcessor):
    """Logits processor that inverts probabilities to tempt rare tokens."""

    def __init__(self, mix: float, chaos_scale: float, eps: float = 1e-6) -> None:
        super().__init__()
        self.mix = mix
        self.chaos_scale = chaos_scale
        self.eps = eps
        self.baseline_probs: List[torch.Tensor] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # VU METER SPOOF: invert probabilities so low-likelihood tokens look loud.
        probs = torch.softmax(scores, dim=-1)
        self.baseline_probs.append(probs.detach().cpu())

        inv = 1.0 / (probs + self.eps)
        q = inv / inv.sum(dim=-1, keepdim=True)
        log_q = torch.log(q)
        bent_scores = log_q * self.chaos_scale
        return (1 - self.mix) * scores + self.mix * bent_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject synthetic perplexity to favor unlikely tokens")
    parser.add_argument("--prompt", required=True, help="Prompt to seed generation")
    parser.add_argument("--chaos_scale", type=float, default=1.0, help="Strength of low-probability boost")
    parser.add_argument("--mix", type=float, default=0.5, help="Blend between baseline and chaos logits [0,1]")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Optional top-k sampling limit")
    parser.add_argument("--model_name", default="gpt2", help="Model name for AutoModelForCausalLM")
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def run_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    processor: Optional[LogitsProcessor] = None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    if top_k is not None:
        gen_kwargs["top_k"] = top_k
    if processor is not None:
        gen_kwargs["logits_processor"] = [processor]
    with torch.no_grad():
        return model.generate(**gen_kwargs)


def compute_ranks(trace_processor: SyntheticPerplexityProcessor, generated: torch.Tensor, prompt_len: int, tokenizer) -> List[StepTrace]:
    traces: List[StepTrace] = []
    gen_tokens = generated[0].tolist()[prompt_len:]
    for step, (token_id, probs) in enumerate(zip(gen_tokens, trace_processor.baseline_probs)):
        # Token rank under original probability distribution (1 = highest prob)
        sorted_ids = torch.argsort(probs[0], descending=True)
        rank = (sorted_ids == token_id).nonzero(as_tuple=False).item() + 1
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        traces.append(StepTrace(token_id=token_id, token_text=token_text, rank_under_p=rank, p_value=float(probs[0, token_id])))
        print(f"Step {step + 1}: chosen token rank under normal = {rank}")
    return traces


def save_run(
    prompt: str,
    chaos_scale: float,
    mix: float,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    baseline_text: str,
    bent_text: str,
    traces: List[StepTrace],
) -> str:
    os.makedirs("runs/synthetic_perplexity_injection", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("runs/synthetic_perplexity_injection", f"run_{timestamp}.json")
    payload = {
        "prompt": prompt,
        "chaos_scale": chaos_scale,
        "mix": mix,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "baseline_text": baseline_text,
        "bent_text": bent_text,
        "trace": [trace.__dict__ for trace in traces],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # BASELINE RUN: no chaos, clean perplexity meter.
    with torch.no_grad():
        baseline_ids = run_generation(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            processor=None,
        )
    baseline_text = tokenizer.decode(baseline_ids[0], skip_special_tokens=True)
    print("Baseline continuation:\n", baseline_text)

    # CHAOS RUN: inverted probabilities lure the sampler toward rare tokens.
    processor = SyntheticPerplexityProcessor(mix=args.mix, chaos_scale=args.chaos_scale)
    bent_ids = run_generation(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        processor=processor,
    )
    bent_text = tokenizer.decode(bent_ids[0], skip_special_tokens=True)
    print("\nBent continuation (synthetic perplexity):\n", bent_text)

    prompt_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    traces = compute_ranks(processor, bent_ids, prompt_len=prompt_ids.shape[1], tokenizer=tokenizer)

    log_path = save_run(
        prompt=args.prompt,
        chaos_scale=args.chaos_scale,
        mix=args.mix,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        baseline_text=baseline_text,
        bent_text=bent_text,
        traces=traces,
    )
    print(f"\nLog saved to {log_path}")
    print("// LIE METER: feeding the sampler a warped perplexity readout to chase aesthetic ghosts.")


if __name__ == "__main__":
    main()
