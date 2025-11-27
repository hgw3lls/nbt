"""
Demonstration of residual bottlenecking on GPT-2 blocks.

This script loads a small GPT-2 model, instruments the residual stream
around attention/MLP sublayers, and replaces selected blocks with
bottlenecked versions that compress the residual signal through a linear
autoencoder.
"""

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ResidualTrace:
    """Container for residual tensors around sublayers."""

    before_attention: torch.Tensor
    after_attention: torch.Tensor
    after_mlp: torch.Tensor


class BottleneckResidualBlock(nn.Module):
    """
    Wraps a GPT-2 block and bottlenecks the residual stream.

    The bottleneck is applied after the attention residual addition and before
    the MLP sublayer. We record reconstruction error to quantify compression.
    """

    def __init__(
        self,
        base_block: GPT2Block,
        d_bottleneck: int = 128,
        activation: Optional[nn.Module] = nn.GELU(),
    ) -> None:
        super().__init__()
        self.base_block = base_block
        d_model = base_block.ln_1.normalized_shape[0]
        self.down = nn.Linear(d_model, d_bottleneck)
        self.up = nn.Linear(d_bottleneck, d_model)
        self.activation = activation
        self.mse = nn.MSELoss(reduction="none")
        self.reconstruction_errors: List[float] = []

    def bottleneck(self, residual: torch.Tensor) -> torch.Tensor:
        compressed = self.down(residual)
        if self.activation is not None:
            compressed = self.activation(compressed)
        return self.up(compressed)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        # --- Residual stream before attention ---
        residual = hidden_states
        hidden_states = self.base_block.ln_1(hidden_states)
        attn_output, self_attn_weights = self.base_block.attn(
            hidden_states,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )

        # --- Residual addition after attention ---
        hidden_states = residual + attn_output

        # --- Bottleneck on residual stream ---
        recon = self.bottleneck(hidden_states)
        # Record MSE averaged over sequence and features.
        per_token_error = self.mse(hidden_states.detach(), recon.detach()).mean(dim=(-1, -2))
        self.reconstruction_errors.extend(per_token_error.flatten().tolist())
        hidden_states = recon

        # Continue with MLP/residual path using the bottlenecked residual stream.
        residual = hidden_states
        hidden_states = self.base_block.ln_2(hidden_states)
        feed_forward_hidden_states = self.base_block.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        outputs: Tuple[torch.Tensor, ...] = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


def load_models(model_name: str = "gpt2"):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return tokenizer, model


def inspect_residual_connections(model: GPT2LMHeadModel) -> ResidualTrace:
    """Runs a single block manually to expose residual tensors."""

    with torch.no_grad():
        encoded = model.transformer.wte(
            torch.tensor([[model.config.bos_token_id or model.config.eos_token_id]], device=device)
        )
        block = model.transformer.h[0]
        residual = encoded
        after_attention = None

        # attention sublayer
        normalized = block.ln_1(encoded)
        attn_output = block.attn(normalized)[0]
        after_attention = residual + attn_output

        # mlp sublayer
        mlp_input = block.ln_2(after_attention)
        after_mlp = after_attention + block.mlp(mlp_input)

    return ResidualTrace(before_attention=residual.cpu(), after_attention=after_attention.cpu(), after_mlp=after_mlp.cpu())


def apply_bottlenecks(model: GPT2LMHeadModel, indices: List[int], d_bottleneck: int = 128) -> GPT2LMHeadModel:
    bottleneck_model = copy.deepcopy(model)
    for idx in indices:
        bottleneck_model.transformer.h[idx] = BottleneckResidualBlock(
            bottleneck_model.transformer.h[idx], d_bottleneck=d_bottleneck
        )
    return bottleneck_model


def generate(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast, prompt: str, max_new_tokens: int = 50) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    tokenizer, base_model = load_models()
    prompts = [
        "The castle stood alone on the hill, its windows glowing as the storm rolled in.",
        "Under the cracked neon sign, the detective waited for the informant.",
    ]

    # Inspect the shapes of the residual stream in the first block.
    trace = inspect_residual_connections(base_model)
    print("Residual tensor shapes (block 0):")
    print(f"  before attention: {tuple(trace.before_attention.shape)}")
    print(f"  after attention:  {tuple(trace.after_attention.shape)}")
    print(f"  after MLP:        {tuple(trace.after_mlp.shape)}")

    # Create a bottlenecked model by wrapping the first two blocks.
    bottleneck_model = apply_bottlenecks(base_model, indices=[0, 1], d_bottleneck=128).to(device)

    print("\nSample generations (original vs. bottlenecked):")
    for prompt in prompts:
        original = generate(base_model, tokenizer, prompt)
        bottlenecked = generate(bottleneck_model, tokenizer, prompt)
        print("\n=== Prompt ===")
        print(prompt)
        print("--- Original ---")
        print(original)
        print("--- Bottlenecked ---")
        print(bottlenecked)

    # Gather reconstruction error statistics.
    errors = []
    for module in bottleneck_model.modules():
        if isinstance(module, BottleneckResidualBlock):
            errors.extend(module.reconstruction_errors)
    if errors:
        errors_tensor = torch.tensor(errors)
        print("\nReconstruction error (MSE) across bottlenecked residuals:")
        print(f"  mean: {errors_tensor.mean().item():.6f}")
        print(f"  std:  {errors_tensor.std().item():.6f}")
    else:
        print("No reconstruction errors recorded.")


if __name__ == "__main__":
    main()
