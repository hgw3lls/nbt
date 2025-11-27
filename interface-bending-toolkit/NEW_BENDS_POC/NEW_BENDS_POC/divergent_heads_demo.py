"""Demonstration of a "divergent heads" bend on GPT-2 attention.

Loads a GPT-style causal LM, replaces one block with a variant that biases
individual heads toward different token regions, and compares generations
against the untouched model. The divergence strength can be tuned via
`focus_factor` below.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention


@dataclass
class DivergenceSpec:
    """Defines which heads focus on which token regions."""

    first_head: int = 0
    middle_head: int = 1
    end_head: int = 2
    focus_factor: float = 1.6  # >1 strengthens bias toward the selected region


class DivergentAttention(torch.nn.Module):
    """Wraps GPT-2 self-attention and reweights attention per head.

    The wrapped module shares parameters with the original GPT-2 attention
    block, but intercepts the computed attention weights to enforce that
    different heads focus on different token regions.
    """

    def __init__(self, base_attn: GPT2Attention, divergence: DivergenceSpec):
        super().__init__()
        self.base_attn = base_attn
        self.divergence = divergence
        self.last_pre_weights: Optional[torch.Tensor] = None
        self.last_post_weights: Optional[torch.Tensor] = None

    def _split_regions(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return boolean masks for beginning, middle, and end token ranges."""

        first_end = max(1, seq_len // 3)
        second_end = max(first_end + 1, 2 * seq_len // 3)
        positions = torch.arange(seq_len)
        first_mask = positions < first_end
        middle_mask = (positions >= first_end) & (positions < second_end)
        end_mask = positions >= second_end
        return first_mask, middle_mask, end_mask

    def _apply_divergence(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """Softly up-weight region-specific tokens for selected heads."""

        # attn_weights: [batch, heads, q_len, k_len]
        bsz, num_heads, _, seq_len = attn_weights.shape
        first_mask, middle_mask, end_mask = self._split_regions(seq_len)
        factor = self.divergence.focus_factor

        # Clone to avoid in-place modifications that could break autograd
        reweighted = attn_weights.clone()
        head_masks = {
            self.divergence.first_head: first_mask,
            self.divergence.middle_head: middle_mask,
            self.divergence.end_head: end_mask,
        }

        for head_idx, region_mask in head_masks.items():
            if head_idx >= num_heads:
                continue  # Skip if the model exposes fewer than 3 heads
            # Expand mask to match attention shape for broadcasting
            region = region_mask.to(attn_weights.device).view(1, 1, 1, seq_len)
            scaled = reweighted[:, head_idx : head_idx + 1] * (
                1 + (factor - 1) * region
            )
            # Renormalize the distribution for that head
            scaled = scaled / scaled.sum(dim=-1, keepdim=True)
            reweighted[:, head_idx : head_idx + 1] = scaled

        return reweighted

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, _ = hidden_states.size()
        qkv = self.base_attn.c_attn(hidden_states)
        query, key, value = qkv.split(self.base_attn.split_size, dim=2)

        def _reshape(x):
            new_shape = (bsz, -1, self.base_attn.num_heads, self.base_attn.head_dim)
            return x.view(new_shape).transpose(1, 2)

        query = _reshape(query)
        key = _reshape(key)
        value = _reshape(value)

        attn_scores = torch.matmul(query, key.transpose(-1, -2))
        if self.base_attn.scale_attn_weights:
            attn_scores = attn_scores / math.sqrt(self.base_attn.head_dim)

        causal_mask = self.base_attn.bias[:, :, :seq_len, :seq_len]
        mask_value = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(~causal_mask, mask_value)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = torch.softmax(attn_scores, dim=-1)
        self.last_pre_weights = attn_weights.detach().cpu()

        attn_weights = self._apply_divergence(attn_weights)
        attn_weights = self.base_attn.attn_dropout(attn_weights)
        self.last_post_weights = attn_weights.detach().cpu()

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, seq_len, -1)
        attn_output = self.base_attn.c_proj(attn_output)
        attn_output = self.base_attn.resid_dropout(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class DivergentBlock(torch.nn.Module):
    """Clone of GPT2Block that swaps in DivergentAttention."""

    def __init__(self, base_block: GPT2Block, divergence: DivergenceSpec):
        super().__init__()
        self.ln_1 = base_block.ln_1
        self.ln_2 = base_block.ln_2
        self.mlp = base_block.mlp
        self.attn = DivergentAttention(base_block.attn, divergence)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        # Cross-attention is unsupported for GPT-2; fall back to base behavior.
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        attn_output = attn_outputs[0]
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states, None)
        if output_attentions:
            outputs = outputs + (attn_outputs[1],)
        return outputs


@torch.no_grad()
def generate_with_model(model, tokenizer, prompt: str, max_new_tokens: int = 40) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def summarize_attention(attn: torch.Tensor, head_idx: int = 0, query_position: int = -1) -> str:
    """Pretty-print a small attention grid for a single head."""

    head_slice = attn[0, head_idx, query_position].numpy()
    return "[" + ", ".join(f"{p:.2f}" for p in head_slice[:12]) + (" ..." if head_slice.size > 12 else "]")


if __name__ == "__main__":
    torch.manual_seed(42)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    base_model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Build divergent model by replacing a single transformer block.
    divergent_spec = DivergenceSpec(focus_factor=1.8)
    divergent_model = GPT2LMHeadModel.from_pretrained("gpt2")
    divergent_model.config.use_cache = False
    target_layer = 1  # Which block to bend
    divergent_model.transformer.h[target_layer] = DivergentBlock(
        divergent_model.transformer.h[target_layer], divergence=divergent_spec
    )

    prompts = [
        "In a tense parliamentary debate, the senator argued that economic reform must balance freedom and equality,",
        "The research paper describes a neural network that learns language while coordinating with a robotics platform,",
        "During the emergency press conference, the mayor acknowledged mistakes but promised transparent recovery efforts,",
    ]

    for prompt in prompts:
        print("Prompt:", prompt)
        print("\nOriginal model ->")
        print(generate_with_model(base_model, tokenizer, prompt))
        print("\nDivergent-heads model ->")
        print(generate_with_model(divergent_model, tokenizer, prompt))
        print("\n" + "=" * 80 + "\n")

    # Run one forward pass with attention outputs to inspect head 0/1/2 biases.
    sample = tokenizer(prompts[0], return_tensors="pt")
    divergent_model(**sample, output_attentions=True)
    attn_before = divergent_model.transformer.h[target_layer].attn.last_pre_weights
    attn_after = divergent_model.transformer.h[target_layer].attn.last_post_weights

    print("Head 0 (beginning-focused) attention before bending:")
    print(summarize_attention(attn_before, head_idx=divergent_spec.first_head))
    print("Head 0 attention after bending:")
    print(summarize_attention(attn_after, head_idx=divergent_spec.first_head))

    print("\nHead 1 (middle-focused) attention before bending:")
    print(summarize_attention(attn_before, head_idx=divergent_spec.middle_head))
    print("Head 1 attention after bending:")
    print(summarize_attention(attn_after, head_idx=divergent_spec.middle_head))

    print("\nHead 2 (end-focused) attention before bending:")
    print(summarize_attention(attn_before, head_idx=divergent_spec.end_head))
    print("Head 2 attention after bending:")
    print(summarize_attention(attn_after, head_idx=divergent_spec.end_head))

    print("\nTune `DivergenceSpec.focus_factor` to make the bend stronger or weaker.")
