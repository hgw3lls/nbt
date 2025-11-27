"""Demo script for justice-oriented temporal anchoring using GPT-2.

This script shows how to bias a transformer's positional encodings around
historical or justice-related reference phrases. The goal is to make the
model more likely to relate contemporary prompts to historical struggles by
"bending" the positional signals for selected tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


@dataclass
class AnchorConfig:
    """Configuration for the temporal anchoring behaviour."""

    alpha: float = 2.0
    neighbor_radius: int = 1
    add_offset: bool = True
    offset_scale: float = 0.5


class JusticeAnchorWrapper(nn.Module):
    """Wraps a GPT-2 model and bends positional encodings for reference tokens."""

    def __init__(
        self,
        base_model: GPT2LMHeadModel,
        tokenizer: GPT2TokenizerFast,
        justice_phrases: Sequence[str],
        anchor_config: AnchorConfig,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.anchor_config = anchor_config

        # Pre-tokenize the justice phrases so we can find them quickly at runtime.
        self.justice_token_sequences: List[Tuple[int, ...]] = [
            tuple(tokenizer.encode(p, add_special_tokens=False)) for p in justice_phrases
        ]

        # Offset vector that will be added to positional embeddings when requested.
        hidden = base_model.config.n_embd
        self.positional_offset = nn.Parameter(torch.randn(hidden) * anchor_config.offset_scale)

        self._current_anchor_mask: torch.Tensor | None = None

        # Register a hook that will be called every time the position embeddings
        # (wpe) are produced. The hook uses ``self._current_anchor_mask``
        # computed in ``forward`` to scale/add offsets to specific positions.
        self.base_model.transformer.wpe.register_forward_hook(self._positional_hook)

    def _locate_anchors(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask of positions that match any justice phrase."""

        # input_ids: (batch, seq_len)
        batch, seq_len = input_ids.shape
        anchor_mask = torch.zeros((batch, seq_len), dtype=torch.bool, device=input_ids.device)

        for seq in self.justice_token_sequences:
            if not seq:
                continue
            k = len(seq)
            # Slide across the sequence and mark any exact matches.
            for start in range(seq_len - k + 1):
                window = input_ids[:, start : start + k]
                matches = (window == torch.tensor(seq, device=input_ids.device)).all(dim=1)
                if matches.any():
                    anchor_mask[matches, start : start + k] = True

        if self.anchor_config.neighbor_radius > 0 and anchor_mask.any():
            radius = self.anchor_config.neighbor_radius
            # Expand influence to neighboring tokens to create a small temporal halo.
            dilated = anchor_mask.clone()
            for shift in range(1, radius + 1):
                dilated[:, shift:] |= anchor_mask[:, :-shift]
                dilated[:, :-shift] |= anchor_mask[:, shift:]
            anchor_mask = dilated

        return anchor_mask

    def _positional_hook(self, module: nn.Module, inputs, output: torch.Tensor) -> torch.Tensor:
        """Bends positional encodings for anchor positions.

        The hook is triggered right after GPT-2's position embeddings (``wpe``)
        are computed. We use ``self._current_anchor_mask`` prepared in ``forward``
        to scale the embeddings (``alpha``) and optionally add a learned offset
        vector so these positions stand out temporally.
        """

        if self._current_anchor_mask is None:
            return output

        mask = self._current_anchor_mask.to(output.device)
        # Scale positional encodings for anchor positions.
        factor = 1.0 + (self.anchor_config.alpha - 1.0) * mask.unsqueeze(-1)
        bent = output * factor

        if self.anchor_config.add_offset:
            bent = bent + mask.unsqueeze(-1) * self.positional_offset

        return bent

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions:
        # Prepare attention mask if not provided.
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Identify which token positions should receive temporal anchoring.
        self._current_anchor_mask = self._locate_anchors(input_ids)

        # Delegate to the base model. The position embedding hook fires inside
        # this call, bending positional encodings for the selected tokens.
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Clear the mask to avoid stale state if the module is reused without a
        # new forward pass.
        self._current_anchor_mask = None
        return output


def iterative_generate(model: nn.Module, tokenizer: GPT2TokenizerFast, prompt: str, max_new_tokens: int = 60) -> str:
    """Simple autoregressive generation that works for both base and wrapped models."""

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def log_token_probabilities(
    model: nn.Module,
    tokenizer: GPT2TokenizerFast,
    prompt: str,
    targets: Iterable[str],
) -> Dict[str, float]:
    """Return token probabilities for the next token given a prompt."""

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

    result: Dict[str, float] = {}
    for t in targets:
        token_id = tokenizer.encode(t, add_special_tokens=False)
        if len(token_id) != 1:
            continue  # multi-token targets are skipped for simplicity
        prob = probs[0, token_id[0]].item()
        result[t] = prob
    return result


def main() -> None:
    # Define justice-oriented reference phrases that should receive temporal anchoring.
    justice_phrases = [
        "civil rights movement",
        "treaty",
        "strike",
        "abolition",
        "land back",
        "freedom riders",
    ]

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    anchor_config = AnchorConfig(alpha=2.5, neighbor_radius=1, add_offset=True, offset_scale=0.3)
    anchored_model = JusticeAnchorWrapper(base_model, tokenizer, justice_phrases, anchor_config)

    prompts = [
        "How should cities rethink policing in 2024?",
        "What are the bold steps on climate change for the next decade?",
        "How do we guarantee housing as a human right?",
    ]

    # Justice-related targets to track probabilities for.
    target_tokens = ["abolition", "treaty", "strike", "movement", "solidarity"]

    for prompt in prompts:
        print("\n=== Prompt ===")
        print(prompt)

        print("\n-- Base model --")
        base_completion = iterative_generate(base_model, tokenizer, prompt)
        print(base_completion)

        base_probs = log_token_probabilities(base_model, tokenizer, prompt, target_tokens)
        print("Token probabilities (base):", base_probs)

        print("\n-- Temporally anchored model --")
        anchored_completion = iterative_generate(anchored_model, tokenizer, prompt)
        print(anchored_completion)

        anchored_probs = log_token_probabilities(anchored_model, tokenizer, prompt, target_tokens)
        print("Token probabilities (anchored):", anchored_probs)

        # Highlight whether anchoring increases justice-related probabilities.
        deltas = {
            t: anchored_probs.get(t, float('nan')) - base_probs.get(t, 0.0)
            for t in target_tokens
        }
        print("Probability deltas (anchored - base):", deltas)


if __name__ == "__main__":
    main()
