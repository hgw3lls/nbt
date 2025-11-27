import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KV memory drift demo: perturb keys more than values to misaddress cached memories."
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to condition generation.")
    parser.add_argument("--key_noise_scale", type=float, default=0.1, help="Stddev of Gaussian noise for keys.")
    parser.add_argument(
        "--value_noise_scale", type=float, default=0.0, help="Stddev of Gaussian noise for values (smaller)."
    )
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Number of tokens to generate.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name for loading GPT-2 variants.")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device."
    )
    return parser.parse_args()


def generate_baseline(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, prompt: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**encoded, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def add_noise_to_past(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    key_noise_scale: float,
    value_noise_scale: float,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    noisy_past = []
    key_l2s: List[float] = []
    for keys, values in past_key_values:
        # KEY LINE: we corrupt addressing more than content, like swapping traces on a memory PCB.
        if key_noise_scale > 0:
            key_noise = torch.randn_like(keys) * key_noise_scale
            keys = keys + key_noise
            key_l2s.append(key_noise.pow(2).mean().sqrt().item())
        else:
            key_l2s.append(0.0)
        if value_noise_scale > 0:
            value_noise = torch.randn_like(values) * value_noise_scale
            values = values + value_noise
        noisy_past.append((keys, values))
    return tuple(noisy_past), float(sum(key_l2s) / len(key_l2s)) if key_l2s else 0.0


def generate_bent(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int,
    key_noise_scale: float,
    value_noise_scale: float,
) -> Tuple[str, List[float]]:
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)["input_ids"]
    generated = input_ids
    past = None
    key_noise_trace: List[float] = []

    for step in range(max_new_tokens):
        input_slice = generated if past is None else generated[:, -1:]
        with torch.no_grad():
            outputs = model(input_ids=input_slice, past_key_values=past, use_cache=True)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            past = outputs.past_key_values

        if past is not None:
            past, key_l2 = add_noise_to_past(past, key_noise_scale, value_noise_scale)
            key_noise_trace.append(key_l2)
        else:
            key_noise_trace.append(0.0)

        generated = torch.cat([generated, next_token], dim=-1)

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text, key_noise_trace


def save_log(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = run_dir / f"kv_memory_drift_{timestamp}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved log to {path}")


def print_trace(noise_trace: List[float]) -> None:
    print("\nNoise trace (avg L2 per step for keys):")
    for idx, l2 in enumerate(noise_trace, start=1):
        bar = "#" * max(1, int(l2 * 50)) if l2 > 0 else "-"
        print(f"Step {idx:03d}: {l2:.4f} {bar}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.to(device)

    # PATCH PAD: baseline run with pristine memory traces.
    print("Running baseline (no drift)...")
    baseline_text = generate_baseline(model, tokenizer, args.prompt, args.max_new_tokens)
    print("\nBaseline output:\n", baseline_text)

    # MEMORY BEND: perturbing keys heavier than values to desync addressing vs content.
    print("\nRunning bent generation with KV drift...")
    bent_text, noise_trace = generate_bent(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.key_noise_scale,
        args.value_noise_scale,
    )
    print("\nBent output:\n", bent_text)
    print_trace(noise_trace)

    log_payload = {
        "prompt": args.prompt,
        "key_noise_scale": args.key_noise_scale,
        "value_noise_scale": args.value_noise_scale,
        "max_new_tokens": args.max_new_tokens,
        "baseline_text": baseline_text,
        "bent_text": bent_text,
        "key_noise_trace_l2": noise_trace,
    }
    save_log(Path("runs/kv_memory_drift"), log_payload)


if __name__ == "__main__":
    main()
