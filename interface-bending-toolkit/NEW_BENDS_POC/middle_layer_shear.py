import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# SHEAR PATCH: twisting mid-stream traces to skew conceptual geometry

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shear middle-layer activations to skew transformer geometry.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation")
    parser.add_argument("--layer_start", type=int, default=8, help="First layer index to shear (inclusive)")
    parser.add_argument("--layer_end", type=int, default=12, help="Last layer index to shear (inclusive)")
    parser.add_argument("--shear_strength", type=float, default=0.3, help="Magnitude of shear transformation")
    parser.add_argument("--mix", type=float, default=0.5, help="Blend between original and sheared activations")
    parser.add_argument("--max_new_tokens", type=int, default=60, help="Number of tokens to generate")
    parser.add_argument("--model_name", type=str, default="gpt2", help="GPT-2 model variant")
    return parser.parse_args()


def load_model_and_tokenizer(model_name: str):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def shear_hook_factory(shear_strength: float, mix: float, layer_idx: int, deltas: Dict[int, List[float]]):
    def hook(module, inputs, output):
        # HEADWAY TWIST: bending the board trace mid-flight
        # This simulates twisting the conceptual geometry mid-stream, like flexing a circuit board to reroute flow.
        hidden_states = output[0] if isinstance(output, tuple) else output
        left, right = torch.chunk(hidden_states, 2, dim=-1)
        right_bent = right + shear_strength * left
        sheared = torch.cat([left, right_bent], dim=-1)
        blended = (1 - mix) * hidden_states + mix * sheared
        delta = (blended - hidden_states).norm(p=2, dim=-1).mean().item()
        deltas[layer_idx].append(delta)
        if isinstance(output, tuple):
            return (blended,) + output[1:]
        return blended

    return hook


def register_shear_hooks(model: GPT2LMHeadModel, layer_start: int, layer_end: int, shear_strength: float, mix: float, deltas: Dict[int, List[float]]):
    handles = []
    for idx in range(layer_start, layer_end + 1):
        if idx < 0 or idx >= len(model.transformer.h):
            raise ValueError(f"Layer index {idx} is out of bounds for this model")
        hook = shear_hook_factory(shear_strength, mix, idx, deltas)
        handles.append(model.transformer.h[idx].register_forward_hook(hook))
    return handles


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


def generate_text(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, prompt: str, max_new_tokens: int) -> str:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def summarize_deltas(deltas: Dict[int, List[float]]) -> Dict[int, float]:
    return {layer: float(torch.tensor(values).mean().item()) if values else 0.0 for layer, values in deltas.items()}


def print_delta_report(avg_deltas: Dict[int, float]):
    print("\n=== Shear Diagnostics ===")
    for layer in sorted(avg_deltas.keys()):
        value = avg_deltas[layer]
        bar = "#" * max(1, int(value * 50))
        print(f"Layer {layer} shear \u0394 = {value:.4f} | {bar}")


def log_run(payload: dict, folder: str = "runs/middle_layer_shear") -> None:
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(folder, f"shear_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Logged run to {path}")


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    print("Running baseline (no shear)...")
    baseline_text = generate_text(model, tokenizer, args.prompt, args.max_new_tokens)

    deltas = defaultdict(list)
    print("Installing shear hooks...")
    handles = register_shear_hooks(
        model,
        args.layer_start,
        args.layer_end,
        args.shear_strength,
        args.mix,
        deltas,
    )

    print("Generating with sheared mid-layers...")
    bent_text = generate_text(model, tokenizer, args.prompt, args.max_new_tokens)
    avg_deltas = summarize_deltas(deltas)
    print_delta_report(avg_deltas)

    remove_hooks(handles)

    payload = {
        "prompt": args.prompt,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
        "shear_strength": args.shear_strength,
        "mix": args.mix,
        "max_new_tokens": args.max_new_tokens,
        "model_name": args.model_name,
        "baseline_text": baseline_text,
        "bent_text": bent_text,
        "layer_deltas": avg_deltas,
    }
    log_run(payload)

    print("\n=== Output Comparison ===")
    print("-- Baseline --")
    print(baseline_text)
    print("\n-- Sheared --")
    print(bent_text)


if __name__ == "__main__":
    main()
