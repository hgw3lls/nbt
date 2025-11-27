import argparse
import random
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ScrambleConfig:
    mode: str = "none"
    span: int = 8
    seed: Optional[int] = None


def describe_positional_addition() -> str:
    """
    Return a short description of where GPT-2 adds positional encodings.

    In transformers' GPT2Model.forward, token embeddings (wte) and positional
    embeddings (wpe) are summed before entering the transformer blocks:
    `hidden_states = inputs_embeds + position_embeds`.
    """

    return (
        "GPT-2 uses learned absolute positional embeddings stored in `wpe` and "
        "adds them to token embeddings (`wte`) inside `GPT2Model.forward` before "
        "passing the result into the attention blocks. By overriding the "
        "`position_ids` argument we can perturb or permute positional indices "
        "without changing the content tokens."
    )


def scramble_position_ids(seq_len: int, config: ScrambleConfig) -> torch.Tensor:
    """Create scrambled position ids according to the chosen mode."""

    if config.mode == "none":
        return torch.arange(seq_len)

    rng = random.Random(config.seed)
    positions: List[int] = []

    if config.mode == "global":
        positions = list(range(seq_len))
        rng.shuffle(positions)
    elif config.mode == "local":
        span = max(1, config.span)
        for start in range(0, seq_len, span):
            block = list(range(start, min(start + span, seq_len)))
            rng.shuffle(block)
            positions.extend(block)
    else:
        raise ValueError(f"Unknown scramble mode: {config.mode}")

    return torch.tensor(positions, dtype=torch.long)


class ScrambledModel:
    def __init__(self, model, config: ScrambleConfig):
        self.model = model
        self.config = config

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Greedy generation using custom position_ids."""

        device = next(self.model.parameters()).device
        generated = input_ids.to(device)
        attn_mask = attention_mask.to(device) if attention_mask is not None else None

        for _ in range(max_new_tokens):
            seq_len = generated.shape[-1]
            position_ids = scramble_position_ids(seq_len, self.config).to(device)

            outputs = self.model(
                input_ids=generated,
                attention_mask=attn_mask,
                position_ids=position_ids,
            )
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, torch.ones_like(next_token, device=device)], dim=-1
                )

        return generated


def decode_tokens(tokenizer, tokens: torch.Tensor) -> str:
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


def visualize_mapping(position_ids: torch.Tensor, title: str, outfile: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(position_ids.cpu().numpy(), label="scrambled index")
    plt.plot(list(range(len(position_ids))), label="original index", linestyle="--")
    plt.xlabel("Token position")
    plt.ylabel("Assigned position id")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    print(f"Saved mapping visualization to {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Temporal scrambling demo")
    parser.add_argument("--prompt", type=str, default=(
        "A courier describes the sequence of events on a long journey through the desert."
    ))
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--mode", choices=["none", "local", "global"], default="local")
    parser.add_argument("--span", type=int, default=8, help="Span size for local scrambling")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--visualize", action="store_true", help="Save a mapping plot for the first run"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(describe_positional_addition())

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded.input_ids
    attention_mask = encoded.attention_mask

    # Baseline generation with natural ordering.
    base_wrapper = ScrambledModel(model, ScrambleConfig(mode="none"))
    base_output = base_wrapper.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
    )

    scrambled_config = ScrambleConfig(mode=args.mode, span=args.span, seed=args.seed)
    scrambled_wrapper = ScrambledModel(model, scrambled_config)
    scrambled_output = scrambled_wrapper.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
    )

    base_text = decode_tokens(tokenizer, base_output)
    scrambled_text = decode_tokens(tokenizer, scrambled_output)

    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Base model continuation ===")
    print(base_text)
    print("\n=== Scrambled positional continuation ===")
    print(scrambled_text)

    if args.visualize:
        example_positions = scramble_position_ids(input_ids.shape[-1], scrambled_config)
        visualize_mapping(
            example_positions,
            title=f"Scramble mode: {args.mode}",
            outfile="scramble_mapping.png",
        )


if __name__ == "__main__":
    main()
