"""Demonstrate a residual re-weighting toward relational framing using GPT-2.

This script loads GPT-2, defines a set of relational tokens, constructs a
residual stream hook that projects activations toward their average embedding,
and compares generations from the base and modified model.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


@dataclass
class RelationalConfig:
    """Configuration options for the relational residual bend."""

    relational_tokens: List[str] = field(
        default_factory=lambda: [
            "system",
            "community",
            "network",
            "infrastructure",
            "history",
            "relations",
            "context",
        ]
    )
    layer_indices: Sequence[int] = field(default_factory=lambda: [5, 8, 11])
    gamma: float = 1.0
    model_name: str = "gpt2"
    max_new_tokens: int = 60
    temperature: float = 0.8
    top_p: float = 0.95
    prompts: List[str] = field(
        default_factory=lambda: [
            "crime in cities",
            "climate change",
            "public health",
        ]
    )


def compute_relational_direction(
    tokenizer: GPT2TokenizerFast, model: GPT2LMHeadModel, tokens: Iterable[str]
) -> torch.Tensor:
    """Average the embeddings of relational tokens to form a direction vector."""

    token_ids = tokenizer(
        list(tokens),
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )["input_ids"]
    embeddings = model.get_input_embeddings()(token_ids)
    direction = embeddings.mean(dim=1).mean(dim=0)  # average over tokens and pieces
    return direction / direction.norm()  # normalize to unit length


def attach_relational_hooks(
    base_model: GPT2LMHeadModel,
    direction: torch.Tensor,
    layer_indices: Sequence[int],
    gamma: float,
) -> GPT2LMHeadModel:
    """Attach forward hooks to a copy of ``base_model`` that nudge residuals."""

    model = copy.deepcopy(base_model)

    def make_hook():
        def hook(module, inputs, output):
            hidden_states = output[0]
            # Project residuals onto the direction and add a scaled version back.
            projection = torch.einsum("bld,d->bl", hidden_states, direction)
            adjusted = hidden_states + gamma * projection.unsqueeze(-1) * direction
            return (adjusted,) + output[1:]

        return hook

    for idx in layer_indices:
        model.transformer.h[idx].register_forward_hook(make_hook())

    return model


def generate(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def highlight_keywords(text: str, keywords: Iterable[str]) -> str:
    highlighted = text
    for word in keywords:
        highlighted = highlighted.replace(word, f"**{word}**")
    return highlighted


def run_demo(cfg: RelationalConfig) -> None:
    tokenizer = GPT2TokenizerFast.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name)

    direction = compute_relational_direction(tokenizer, model, cfg.relational_tokens)
    relational_model = attach_relational_hooks(
        model, direction, cfg.layer_indices, cfg.gamma
    )

    individual_markers = ["responsibility", "choice", "individual", "personal"]
    systemic_markers = ["system", "community", "infrastructure", "policy"]

    for prompt in cfg.prompts:
        base_out = generate(
            model,
            tokenizer,
            prompt,
            cfg.max_new_tokens,
            cfg.temperature,
            cfg.top_p,
        )
        relational_out = generate(
            relational_model,
            tokenizer,
            prompt,
            cfg.max_new_tokens,
            cfg.temperature,
            cfg.top_p,
        )

        print("\n=== Prompt ===")
        print(prompt)
        print("--- Base ---")
        print(highlight_keywords(base_out, individual_markers))
        print("--- Relational ---")
        print(highlight_keywords(relational_out, systemic_markers))


def parse_args() -> RelationalConfig:
    parser = argparse.ArgumentParser(
        description="Residual re-weighting toward relational framing"
    )
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="Strength of the relational push"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[5, 8, 11],
        help="Transformer block indices to modify",
    )
    parser.add_argument(
        "--relational-tokens",
        nargs="+",
        default=[
            "system",
            "community",
            "network",
            "infrastructure",
            "history",
            "relations",
            "context",
        ],
        help="Tokens used to define the relational direction",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Optional prompt(s) to override the defaults",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=60, help="Tokens to sample"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95, help="Nucleus sampling threshold"
    )

    args = parser.parse_args()

    prompts = args.prompts if args.prompts is not None else None

    return RelationalConfig(
        relational_tokens=args.relational_tokens,
        layer_indices=args.layers,
        gamma=args.gamma,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        prompts=prompts or RelationalConfig().prompts,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_demo(cfg)
