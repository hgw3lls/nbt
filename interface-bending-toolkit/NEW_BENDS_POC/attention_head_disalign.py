"""
Simulate disagreement among transformer attention head groups via logit bending.

Baseline logits are blended with two competing head-group biases to evoke
circuit-bent divergence between attention clusters.
"""

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor


@dataclass
class StepTrace:
    step: int
    mass_a: float
    mass_b: float


class DisalignProcessor(LogitsProcessor):
    """Logits processor that injects conflicting head-group biases."""

    def __init__(
        self,
        focus_a: Sequence[int],
        focus_b: Sequence[int],
        scale_a: float,
        scale_b: float,
        mix: float,
    ) -> None:
        super().__init__()
        self.focus_a = torch.tensor(focus_a, dtype=torch.long)
        self.focus_b = torch.tensor(focus_b, dtype=torch.long)
        self.scale_a = scale_a
        self.scale_b = scale_b
        self.mix = mix
        self.traces: List[StepTrace] = []

    def to(self, device: torch.device) -> "DisalignProcessor":
        self.focus_a = self.focus_a.to(device)
        self.focus_b = self.focus_b.to(device)
        return self

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # HEAD BUS A: patched to alternative ground; HEAD BUS B: cross-patching elsewhere.
        baseline_logits = scores
        logits_a = scores.clone()
        logits_b = scores.clone()

        if self.focus_a.numel() > 0:
            logits_a[..., self.focus_a] += self.scale_a
        if self.focus_b.numel() > 0:
            logits_b[..., self.focus_b] += self.scale_b

        # Introducing controlled disagreement between head clusters.
        bent_logits = 0.5 * (logits_a + logits_b)
        final_logits = (1 - self.mix) * baseline_logits + self.mix * bent_logits

        with torch.no_grad():
            probs = torch.softmax(final_logits, dim=-1)
            mass_a = probs[..., self.focus_a].sum().item() if self.focus_a.numel() > 0 else 0.0
            mass_b = probs[..., self.focus_b].sum().item() if self.focus_b.numel() > 0 else 0.0
            self.traces.append(StepTrace(step=len(self.traces) + 1, mass_a=mass_a, mass_b=mass_b))

        return final_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Circuit-bent disagreement across attention head groups")
    parser.add_argument("--prompt", required=True, help="Input prompt to condition on")
    parser.add_argument("--focus_tokens_A", required=True, help="Comma-separated tokens for head group A")
    parser.add_argument("--focus_tokens_B", required=True, help="Comma-separated tokens for head group B")
    parser.add_argument("--scale_A", type=float, default=5.0, help="Bias strength for group A")
    parser.add_argument("--scale_B", type=float, default=5.0, help="Bias strength for group B")
    parser.add_argument("--mix", type=float, default=0.7, help="Mix between baseline and bent logits [0,1]")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Length of generation")
    parser.add_argument("--model_name", default="gpt2", help="Model to use for generation")
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
        if encoded:
            ids.extend(encoded)
    return ids


def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def generate_bent(
    model,
    tokenizer,
    prompt: str,
    processor: DisalignProcessor,
    max_new_tokens: int,
) -> str:
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
    return tokenizer.decode(output[0], skip_special_tokens=True)


def format_trace(traces: Sequence[StepTrace]) -> str:
    lines = []
    for t in traces:
        bar_a = "#" * int(round(t.mass_a * 20))
        bar_b = "#" * int(round(t.mass_b * 20))
        lines.append(f"Step {t.step}: A: {bar_a or '.'} B: {bar_b or '.'}")
    return "\n".join(lines)


def log_run(
    log_dir: str,
    args: argparse.Namespace,
    focus_a: Sequence[str],
    focus_b: Sequence[str],
    baseline_text: str,
    bent_text: str,
    traces: Sequence[StepTrace],
) -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(log_dir, f"attention_head_disalign_{timestamp}.json")
    payload = {
        "prompt": args.prompt,
        "focus_tokens_A": list(focus_a),
        "focus_tokens_B": list(focus_b),
        "scale_A": args.scale_A,
        "scale_B": args.scale_B,
        "mix": args.mix,
        "baseline_text": baseline_text,
        "bent_text": bent_text,
        "traces": [trace.__dict__ for trace in traces],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    focus_a_tokens = parse_tokens(args.focus_tokens_A)
    focus_b_tokens = parse_tokens(args.focus_tokens_B)
    focus_a_ids = tokens_to_ids(tokenizer, focus_a_tokens)
    focus_b_ids = tokens_to_ids(tokenizer, focus_b_tokens)

    baseline_text = generate_baseline(model, tokenizer, args.prompt, args.max_new_tokens)
    processor = DisalignProcessor(focus_a_ids, focus_b_ids, args.scale_A, args.scale_B, args.mix)
    bent_text = generate_bent(model, tokenizer, args.prompt, processor, args.max_new_tokens)

    trace_view = format_trace(processor.traces)
    print("=== Baseline ===")
    print(baseline_text)
    print("\n=== Bent (head disalignment) ===")
    print(bent_text)
    print("\n=== Head disagreement trace ===")
    print(trace_view)

    log_path = log_run(
        log_dir=os.path.join("runs", "attention_head_disalign"),
        args=args,
        focus_a=focus_a_tokens,
        focus_b=focus_b_tokens,
        baseline_text=baseline_text,
        bent_text=bent_text,
        traces=processor.traces,
    )
    print(f"Logged run to {log_path}")


if __name__ == "__main__":
    main()
