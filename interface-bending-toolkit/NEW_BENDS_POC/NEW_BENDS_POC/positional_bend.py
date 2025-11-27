"""positional_bend.py
Simulate circuit-bending positional cues by scrambling prompts or anchoring with historical phrases.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor


# CLOCK BEND: miswiring the token sequence to test synthetic time.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal scramble + justice anchoring bend")
    parser.add_argument("--prompt", required=True, help="Input prompt to bend")
    parser.add_argument("--mode", choices=["scramble", "anchor"], required=True)
    parser.add_argument("--anchor_phrases", default="", help="Comma-separated historical anchor phrases")
    parser.add_argument("--scramble_prob", type=float, default=0.5, help="Probability to permute a clause")
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--mix", type=float, default=1.0, help="Dry/wet mix between base and bent logits")
    parser.add_argument("--model_name", default="gpt2", help="HF model name (causal LM)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def sentence_chunks(prompt: str) -> List[str]:
    # Simple sentence splitting that keeps punctuation attached.
    parts: List[str] = []
    start = 0
    for idx, ch in enumerate(prompt):
        if ch in ".!?" and (idx == len(prompt) - 1 or prompt[idx + 1] == " "):
            parts.append(prompt[start : idx + 1].strip())
            start = idx + 1
    if start < len(prompt):
        remainder = prompt[start:].strip()
        if remainder:
            parts.append(remainder)
    return [p for p in parts if p]


def contains_anchor_token(text: str) -> bool:
    return any(char.isdigit() for char in text)


def scramble_prompt(sentences: Sequence[str], prob: float) -> Tuple[List[str], List[int]]:
    indices = list(range(len(sentences)))
    movable = [i for i, sent in enumerate(sentences) if not contains_anchor_token(sent) and random.random() < prob]
    permutable = movable[:]
    random.shuffle(permutable)
    mapping = {src: dst for src, dst in zip(movable, permutable)}
    scrambled: List[str] = []
    permutation_record: List[int] = []
    for i in indices:
        target_idx = mapping.get(i, i)
        permutation_record.append(target_idx)
    for i in sorted(indices, key=lambda x: permutation_record[x]):
        scrambled.append(sentences[i])
    return scrambled, permutation_record


def anchor_prompt(prompt: str, anchors: Sequence[str]) -> Tuple[str, List[str]]:
    if not anchors:
        anchors = ["long history of collective care", "freedom struggles"]
    lead = f"Before we answer, recall the {anchors[0]}. "
    sentences = sentence_chunks(prompt)
    interleaved: List[str] = []
    for idx, sent in enumerate(sentences):
        interleaved.append(sent)
        if idx < len(anchors):
            interleaved.append(anchors[idx])
    anchored_prompt = lead + " ".join(interleaved)
    return anchored_prompt, list(anchors)


class AnchorLogits(LogitsProcessor):
    def __init__(self, tokenizer: AutoTokenizer, anchor_phrases: Sequence[str], mix: float, mode: str):
        self.tokenizer = tokenizer
        self.anchor_ids = self._collect_anchor_ids(anchor_phrases)
        self.mix = mix
        self.mode = mode

    def _collect_anchor_ids(self, phrases: Sequence[str]) -> List[int]:
        ids = []
        for phrase in phrases:
            if not phrase:
                continue
            ids.extend(self.tokenizer.encode(phrase, add_special_tokens=False))
        return list(set(ids))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        base = scores
        bent = scores.clone()
        if self.anchor_ids:
            bias = torch.zeros_like(scores)
            for idx in self.anchor_ids:
                if idx < bias.shape[-1]:
                    bias[..., idx] += 3.0
            bent = bent + bias
        if self.mode == "scramble":
            noise = torch.randn_like(scores) * 0.5
            bent = bent + noise
        return (1 - self.mix) * base + self.mix * bent


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to("cpu")
    return tokenizer, model


def generate_text(prompt: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, max_new_tokens: int, logits_processor=None) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def format_before_after(original: Sequence[str], modified: Sequence[str]) -> str:
    lines = ["Original order:"]
    for idx, sent in enumerate(original):
        lines.append(f"  {idx:02d}: {sent}")
    lines.append("Bent order:")
    for idx, sent in enumerate(modified):
        lines.append(f"  {idx:02d}: {sent}")
    return "\n".join(lines)


def log_run(data: dict, mode: str) -> Path:
    run_dir = Path("runs") / "positional_bend"
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = run_dir / f"{mode}_{timestamp}.json"
    out_path.write_text(json.dumps(data, indent=2))
    return out_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    anchors = [a.strip() for a in args.anchor_phrases.split(",") if a.strip()]
    tokenizer, model = load_model(args.model_name)

    base_sentences = sentence_chunks(args.prompt)
    modified_prompt = args.prompt
    permutation: List[int] = list(range(len(base_sentences)))
    anchor_logit_phrases: List[str] = anchors

    if args.mode == "scramble":
        scrambled, permutation = scramble_prompt(base_sentences, args.scramble_prob)
        modified_prompt = " ".join(scrambled)
        anchor_logit_phrases = scrambled  # bias toward scrambled phrases to reinforce the misclock.
    elif args.mode == "anchor":
        modified_prompt, anchor_logit_phrases = anchor_prompt(args.prompt, anchors)

    # Baseline generation on the unmodified prompt.
    baseline_output = generate_text(
        args.prompt, tokenizer, model, args.max_new_tokens, logits_processor=None
    )

    # Bent generation with positional perturbation and decode-time bias.
    processor = AnchorLogits(tokenizer, anchor_logit_phrases, args.mix, args.mode)
    bent_output = generate_text(
        modified_prompt, tokenizer, model, args.max_new_tokens, logits_processor=[processor]
    )

    before_after = format_before_after(base_sentences, sentence_chunks(modified_prompt))
    print(before_after)
    print("\n=== BASELINE ===\n", baseline_output)
    print("\n=== BENT ===\n", bent_output)

    record = {
        "prompt": args.prompt,
        "mode": args.mode,
        "anchor_phrases": anchors,
        "scramble_prob": args.scramble_prob,
        "mix": args.mix,
        "permutation": permutation,
        "modified_prompt": modified_prompt,
        "baseline_output": baseline_output,
        "bent_output": bent_output,
        "before_after": before_after,
    }
    out_path = log_run(record, args.mode)
    print(f"Logged run to {out_path}")


if __name__ == "__main__":
    main()
