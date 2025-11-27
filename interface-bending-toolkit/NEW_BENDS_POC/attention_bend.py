"""attention_bend.py
A circuit-bent attention-like patch that biases token visibility using logits processors.
"""
import argparse
import json
import os
from collections import Counter
from datetime import datetime
from typing import Iterable, List, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    set_seed,
)


# SHORT: forcing certain semantic nodes to be over-amplified
class BendingLogitsProcessor(LogitsProcessor):
    """Simulate attention bending by nudging logits for specific tokens."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        boost_ids: Iterable[int],
        suppress_ids: Iterable[int],
        boost_scale: float,
        suppress_scale: float,
        mode: str,
        mix: float,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.boost_ids = list(boost_ids)
        self.suppress_ids = list(suppress_ids)
        self.boost_scale = boost_scale
        self.suppress_scale = suppress_scale
        self.mode = mode
        self.mix = mix
        self.step_logs: List[dict] = []
        self.gain_counter: Counter[str] = Counter()

    def _make_delta(self, ids_to_boost: Iterable[int], ids_to_suppress: Iterable[int]) -> torch.Tensor:
        # PATCH POINT: boosting divergent heads in logit space
        delta = torch.zeros(1, self.vocab_size, device=self.device)
        if ids_to_boost:
            delta[:, ids_to_boost] += self.boost_scale
        if ids_to_suppress:
            delta[:, ids_to_suppress] += self.suppress_scale
        return delta

    def _log_step(self, ids_boosted: Iterable[int], ids_suppressed: Iterable[int], stream: str) -> None:
        boosted_tokens = [self.tokenizer.convert_ids_to_tokens(i) for i in ids_boosted]
        suppressed_tokens = [self.tokenizer.convert_ids_to_tokens(i) for i in ids_suppressed]
        for tok in boosted_tokens:
            self.gain_counter[tok] += 1
        self.step_logs.append(
            {
                "step": len(self.step_logs),
                "stream": stream,
                "boosted": boosted_tokens,
                "suppressed": suppressed_tokens,
            }
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:  # type: ignore[override]
        self.device = scores.device
        self.vocab_size = scores.shape[-1]
        base_scores = scores

        if self.mode == "boost":
            delta = self._make_delta(self.boost_ids, self.suppress_ids)
            self._log_step(self.boost_ids, self.suppress_ids, stream="boost")
            bent_scores = base_scores + delta
        elif self.mode == "divergent":
            delta_a = self._make_delta(self.boost_ids, self.suppress_ids)
            delta_b = self._make_delta(self.suppress_ids, self.boost_ids)
            self._log_step(self.boost_ids, self.suppress_ids, stream="A")
            self._log_step(self.suppress_ids, self.boost_ids, stream="B")
            bent_scores = base_scores + 0.5 * (delta_a + delta_b)
        else:
            bent_scores = base_scores

        # Dry/wet mix to emulate patching into the relevance bus
        mixed_scores = (1.0 - self.mix) * base_scores + self.mix * bent_scores
        return mixed_scores


def parse_tokens(tokenizer: AutoTokenizer, token_csv: str) -> Tuple[List[str], List[int]]:
    tokens = [tok.strip() for tok in token_csv.split(",") if tok.strip()]
    ids: List[int] = []
    for tok in tokens:
        encoding = tokenizer.encode(tok, add_special_tokens=False)
        if not encoding:
            continue
        ids.append(encoding[0])
    return tokens, ids


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, logits_processor=None) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generation[0], skip_special_tokens=True)


def save_log(log_data: dict, run_dir: str) -> str:
    os.makedirs(run_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(run_dir, f"attention_bend_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    return path


def print_gain_map(counter: Counter[str]) -> None:
    print("\nGain map (how often tokens were elevated):")
    for token, count in counter.items():
        bar = "▇" * max(1, count)
        print(f"  {token:>12}: {bar} ({count})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Circuit-bend attention via logits tweaks")
    parser.add_argument("--model_name", default="gpt2", help="Model name for AutoModelForCausalLM")
    parser.add_argument("--prompt", default="Patch the relevance bus.", help="Prompt to generate from")
    parser.add_argument("--boost_tokens", default="vision,care", help="Tokens to boost")
    parser.add_argument("--suppress_tokens", default="police,violence", help="Tokens to suppress")
    parser.add_argument("--boost_scale", type=float, default=5.0, help="Logit boost scale")
    parser.add_argument("--suppress_scale", type=float, default=-5.0, help="Logit suppression scale")
    parser.add_argument("--mode", choices=["boost", "divergent"], default="boost", help="Bend mode")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Number of new tokens to generate")
    parser.add_argument("--mix", type=float, default=0.5, help="Dry/wet mix between baseline and bent scores")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    boost_tokens, boost_ids = parse_tokens(tokenizer, args.boost_tokens)
    suppress_tokens, suppress_ids = parse_tokens(tokenizer, args.suppress_tokens)

    print("Loaded model", args.model_name)
    print("Boost tokens:", boost_tokens)
    print("Suppress tokens:", suppress_tokens)

    baseline_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    processor = BendingLogitsProcessor(
        tokenizer=tokenizer,
        boost_ids=boost_ids,
        suppress_ids=suppress_ids,
        boost_scale=args.boost_scale,
        suppress_scale=args.suppress_scale,
        mode=args.mode,
        mix=args.mix,
    )
    logits_processors = LogitsProcessorList([processor])
    bent_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        logits_processor=logits_processors,
    )

    log_data = {
        "prompt": args.prompt,
        "mode": args.mode,
        "boost_tokens": boost_tokens,
        "suppress_tokens": suppress_tokens,
        "boost_scale": args.boost_scale,
        "suppress_scale": args.suppress_scale,
        "mix": args.mix,
        "max_new_tokens": args.max_new_tokens,
        "baseline_output": baseline_text,
        "bent_output": bent_text,
        "step_logs": processor.step_logs,
        "gain_map": processor.gain_counter,
    }

    log_path = save_log(log_data, run_dir=os.path.join("runs", "attention_bend"))

    print("\nBaseline output:\n", baseline_text)
    print("\nBent output:\n", bent_text)
    print(f"\nRun log saved to {log_path}")
    print_gain_map(processor.gain_counter)


if __name__ == "__main__":
    main()
