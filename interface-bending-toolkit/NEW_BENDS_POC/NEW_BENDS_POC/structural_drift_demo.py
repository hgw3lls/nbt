"""
Demonstration of a "substrate drift toward structural analysis" bend for GPT-style transformers.

The script:
1. Builds a structural analysis lexicon.
2. Computes a direction vector using the mean embedding of the lexicon.
3. Registers forward hooks on every decoder block to nudge hidden states toward that direction.
4. Compares generations from the base model and the drifted model on socio-political prompts.

Usage: `python structural_drift_demo.py`
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import List

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, set_seed


@dataclass
class DriftConfig:
    """Configuration for the structural drift bend."""

    eta: float = 0.10  # Magnitude of the drift. Increase/decrease to strengthen/soften the effect.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: str = "gpt2"
    max_new_tokens: int = 80
    temperature: float = 0.8

    # Core vocabulary that defines the structural/systemic direction.
    structural_lexicon: List[str] = (
        "infrastructure",
        "policy",
        "systemic",
        "historical",
        "structural",
        "economy",
        "institution",
    )

    # Simple individualizing vocabulary for rough contrastive counting.
    non_structural_lexicon: List[str] = (
        "individual",
        "personal",
        "choice",
        "character",
        "behavior",
        "self",
        "responsibility",
    )


class StructuralDrift:
    """Implements the hidden-state drift toward structural vocabulary."""

    def __init__(self, config: DriftConfig):
        self.config = config
        self.tokenizer = GPT2TokenizerFast.from_pretrained(config.model_name)
        self.base_model = GPT2LMHeadModel.from_pretrained(config.model_name).to(
            config.device
        )
        # Create an independent copy for the structural model so hooks do not affect the base.
        self.structural_model = GPT2LMHeadModel.from_pretrained(config.model_name).to(
            config.device
        )
        self.direction = self._compute_structural_direction()
        self._register_hooks()

    def _compute_structural_direction(self) -> torch.Tensor:
        """Compute the mean embedding for the structural lexicon as the drift direction.

        The vector is normalized so that scaling is controlled entirely by `eta`.
        Expand the lexicon or swap in hidden states for other layers to shape the direction.
        """

        token_ids = []
        for word in self.config.structural_lexicon:
            encoded = self.tokenizer.encode(word, add_special_tokens=False)
            token_ids.extend(encoded)
        if not token_ids:
            raise ValueError("Structural lexicon produced no token ids.")

        with torch.no_grad():
            embeddings = self.structural_model.transformer.wte(torch.tensor(token_ids)).to(
                self.config.device
            )
            direction = embeddings.mean(dim=0)
            direction = direction / direction.norm()  # Normalize for stability.
        return direction

    def _hook_fn(self, module, inputs, output):
        """Add a small drift toward the structural direction to hidden states.

        GPT-2 block outputs a tuple where the first element is the hidden states.
        """

        hidden_states = output[0]
        drift = self.config.eta * self.direction
        drifted = hidden_states + drift
        return (drifted,) + output[1:]

    def _register_hooks(self) -> None:
        """Attach drift hooks to every decoder block.

        To target specific layers, slice `self.structural_model.transformer.h` before
        registering hooks.
        """

        for block in self.structural_model.transformer.h:
            block.register_forward_hook(self._hook_fn)

    @torch.no_grad()
    def generate(self, prompt: str, use_structural: bool) -> str:
        """Generate text using the base or structural model."""

        model = self.structural_model if use_structural else self.base_model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def count_terms(self, text: str) -> tuple[int, int]:
        """Count occurrences of structural vs non-structural lexicon entries."""

        def count_hits(lexicon: List[str]) -> int:
            pattern = r"|".join(re.escape(w.lower()) for w in lexicon)
            return len(re.findall(pattern, text.lower())) if pattern else 0

        return (
            count_hits(self.config.structural_lexicon),
            count_hits(self.config.non_structural_lexicon),
        )


def main():
    parser = argparse.ArgumentParser(description="Structural drift demonstration")
    parser.add_argument("--eta", type=float, default=None, help="Drift strength (default: 0.10)")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic decoding"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    config = DriftConfig()
    if args.eta is not None:
        config.eta = args.eta

    drift = StructuralDrift(config)

    prompts = [
        "Discuss the causes of housing inequality in major cities.",
        "Why do some communities face higher unemployment rates?",
        "Explain differences in educational outcomes across neighborhoods.",
    ]

    for prompt in prompts:
        print("\n=== Prompt ===")
        print(prompt)

        base_text = drift.generate(prompt, use_structural=False)
        struct_text = drift.generate(prompt, use_structural=True)

        base_counts = drift.count_terms(base_text)
        struct_counts = drift.count_terms(struct_text)

        print("\n--- Base Model Output ---")
        print(base_text)
        print(
            f"Structural terms: {base_counts[0]} | Non-structural terms: {base_counts[1]}"
        )

        print("\n--- Structural Drift Output ---")
        print(struct_text)
        print(
            f"Structural terms: {struct_counts[0]} | Non-structural terms: {struct_counts[1]}"
        )

    print(
        "\nTip: Increase --eta to intensify structural framing, and extend the structural lexicon\n"
        "with additional systemic vocabulary to steer the direction further."
    )


if __name__ == "__main__":
    main()
