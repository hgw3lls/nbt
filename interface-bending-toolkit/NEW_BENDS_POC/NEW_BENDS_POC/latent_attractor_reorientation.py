import argparse
from dataclasses import dataclass
from typing import List

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent attractor reorientation demo for GPT-2")
    parser.add_argument(
        "--bend",
        action="store_true",
        help="Apply the attractor bend. If omitted, runs the base model only.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=6,
        help="Transformer block index at which to inject the attractor direction (0-indexed).",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.25,
        help="Scaling factor for the attractor direction bias.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=40,
        help="Maximum number of tokens to generate (including the prompt).",
    )
    return parser.parse_args()


@dataclass
class AttractorVectors:
    directions: List[torch.Tensor]

    @property
    def mean_direction(self) -> torch.Tensor:
        stacked = torch.stack(self.directions, dim=0)
        return stacked.mean(dim=0)


ATTRACTOR_PHRASES = [
    "we thrive together as a living network",
    "care ripples through every relationship",
    "resources circulate like water in a watershed",
    "decisions arise from listening to the whole",
    "technology roots in reciprocity with the earth",
]


def compute_attractor_vectors(
    tokenizer: GPT2TokenizerFast,
    model: GPT2LMHeadModel,
    layer_index: int,
) -> AttractorVectors:
    """Encode attractor phrases and collect average hidden states from a given layer."""

    vectors: List[torch.Tensor] = []
    device = next(model.parameters()).device

    for phrase in ATTRACTOR_PHRASES:
        encoded = tokenizer(phrase, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_index].squeeze(0)
        vec = hidden.mean(dim=0)  # average across tokens
        vectors.append(vec)

    return AttractorVectors(vectors)


def register_bend_hook(
    model: GPT2LMHeadModel,
    layer_index: int,
    bias_vector: torch.Tensor,
    lmbda: float,
):
    bias = bias_vector * lmbda

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden_state = output[0] + bias
            return (hidden_state, *output[1:])
        return output + bias

    return model.transformer.h[layer_index].register_forward_hook(hook)


def generate_text(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompts: List[str],
    max_length: int,
) -> List[str]:
    device = next(model.parameters()).device
    outputs: List[str] = []

    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_length=max_length,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )
        outputs.append(tokenizer.decode(generated[0], skip_special_tokens=True))
    return outputs


def main():
    args = parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.manual_seed(42)

    prompts = [
        "The future of technology is",
        "A successful life means",
        "Community resilience grows when",
        "Innovation starts with",
    ]

    print("Computing attractor directions...")
    attractors = compute_attractor_vectors(tokenizer, model, args.layer)
    bias_vector = attractors.mean_direction
    bias_vector = bias_vector.to(device)

    print("Generating with base model...\n")
    base_outputs = generate_text(model, tokenizer, prompts, args.max_length)

    hook = None
    bent_outputs: List[str] = []
    if args.bend:
        print("Applying attractor bend at layer", args.layer)
        hook = register_bend_hook(model, args.layer, bias_vector, args.lmbda)
        bent_outputs = generate_text(model, tokenizer, prompts, args.max_length)
        hook.remove()

    for i, prompt in enumerate(prompts):
        print("=== Prompt ===")
        print(prompt)
        print("-- Base --")
        print(base_outputs[i])
        if args.bend:
            print("-- Attractor-biased --")
            print(bent_outputs[i])
        print()


if __name__ == "__main__":
    main()
