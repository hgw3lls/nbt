"""
Subaltern attention boosting demo.
This script loads a masked language model and applies a custom attention hook
that amplifies attention toward user-defined marginalized keywords.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.models.bert.modeling_bert import BertSelfAttention

# ---------------------------
# Configuration flags
# ---------------------------
MODEL_NAME = "bert-base-uncased"
KEYWORDS: List[str] = [
    "mutual aid",
    "abolition",
    "Indigenous knowledge",
    "care work",
    "disability justice",
    "community land trust",
    "restorative justice",
]
BOOST_FACTOR = 1.75
# Set to None for all layers or provide explicit indices (0-indexed).
LAYERS_TO_BOOST: Optional[Sequence[int]] = None
TOP_K = 5  # number of predictions to show for each [MASK]


@dataclass
class BoostStats:
    layer_index: int
    pre_values: List[float] = field(default_factory=list)
    post_values: List[float] = field(default_factory=list)

    def update(self, pre: torch.Tensor, post: torch.Tensor) -> None:
        if pre.numel() == 0:
            return
        self.pre_values.extend(pre.flatten().tolist())
        self.post_values.extend(post.flatten().tolist())

    def summary(self) -> str:
        if not self.pre_values:
            return "    no subaltern tokens seen"
        pre_min, pre_max = min(self.pre_values), max(self.pre_values)
        post_min, post_max = min(self.post_values), max(self.post_values)
        return (
            f"    pre-boost attention range: {pre_min:.4f}–{pre_max:.4f}\n"
            f"    post-boost attention range: {post_min:.4f}–{post_max:.4f}"
        )


class BoostRecorder:
    def __init__(self, num_layers: int):
        self.stats: Dict[int, BoostStats] = {i: BoostStats(i) for i in range(num_layers)}

    def record(self, layer_idx: int, pre: torch.Tensor, post: torch.Tensor) -> None:
        self.stats[layer_idx].update(pre, post)

    def report(self) -> str:
        lines = ["\nAttention changes (subaltern tokens only):"]
        for idx in sorted(self.stats):
            lines.append(f"  Layer {idx}:")
            lines.append(self.stats[idx].summary())
        return "\n".join(lines)


def tokenize_keywords(tokenizer, keywords: Iterable[str]) -> List[List[int]]:
    return [tokenizer.encode(k, add_special_tokens=False) for k in keywords]


def find_subaltern_positions(
    input_ids: torch.Tensor, keyword_token_ids: List[List[int]]
) -> torch.Tensor:
    """Return a boolean mask of shape (batch, seq_len) for keyword occurrences."""
    batch, seq_len = input_ids.shape
    mask = torch.zeros((batch, seq_len), dtype=torch.bool, device=input_ids.device)
    for kw_tokens in keyword_token_ids:
        if not kw_tokens:
            continue
        k = len(kw_tokens)
        for b in range(batch):
            seq = input_ids[b].tolist()
            for i in range(seq_len - k + 1):
                if seq[i : i + k] == kw_tokens:
                    mask[b, i : i + k] = True
    return mask


class BoostedSelfAttention(BertSelfAttention):
    def __init__(
        self,
        original: torch.nn.Module,
        config,
        layer_index: int,
        beta: float,
        get_mask: Callable[[], Optional[torch.Tensor]],
        recorder: BoostRecorder,
    ) -> None:
        super().__init__(config)
        self.load_state_dict(original.state_dict())
        self.beta = beta
        self.get_mask = get_mask
        self.layer_index = layer_index
        self.recorder = recorder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if encoder_hidden_states is not None:
            raise ValueError("BoostedSelfAttention only supports self-attention models.")

        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / self.attention_head_size ** 0.5
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        sub_mask = self.get_mask()
        if sub_mask is not None and sub_mask.any():
            boost_mask = sub_mask[:, None, None, :].to(attention_probs.dtype)
            original_probs = attention_probs.detach().clone()
            attention_probs = attention_probs * (1 + (self.beta - 1) * boost_mask)
            attention_probs = attention_probs / attention_probs.sum(-1, keepdim=True).clamp_min(1e-9)
            value_mask = boost_mask.expand(
                attention_probs.size(0), attention_probs.size(1), attention_probs.size(2), attention_probs.size(3)
            )
            boosted_values = attention_probs[value_mask.bool()]
            original_values = original_probs[value_mask.bool()]
            self.recorder.record(self.layer_index, original_values.cpu(), boosted_values.cpu())

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if use_cache:
            outputs = outputs + ((key_layer, value_layer),)
        return outputs


def apply_attention_boost(
    model: AutoModelForMaskedLM,
    beta: float,
    layer_indices: Optional[Sequence[int]],
    mask_getter: Callable[[], Optional[torch.Tensor]],
    recorder: BoostRecorder,
):
    bert_encoder_layers = model.bert.encoder.layer
    target_layers = range(len(bert_encoder_layers)) if layer_indices is None else layer_indices
    for idx in target_layers:
        bert_layer = bert_encoder_layers[idx]
        bert_layer.attention.self = BoostedSelfAttention(
            bert_layer.attention.self,
            config=model.config,
            layer_index=idx,
            beta=beta,
            get_mask=mask_getter,
            recorder=recorder,
        )


def top_predictions_for_masks(logits: torch.Tensor, input_ids: torch.Tensor, tokenizer, top_k: int) -> List[List[str]]:
    results: List[List[str]] = []
    mask_token_id = tokenizer.mask_token_id
    batch_size = input_ids.size(0)
    vocab_logits = logits.detach().cpu()
    for b in range(batch_size):
        positions = (input_ids[b] == mask_token_id).nonzero(as_tuple=False).flatten().tolist()
        for pos in positions:
            top = torch.topk(vocab_logits[b, pos], k=top_k)
            tokens = [tokenizer.convert_ids_to_tokens(idx) for idx in top.indices.tolist()]
            results.append(tokens)
    return results


def run_comparison():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    boosted_model = deepcopy(model)
    model.eval()
    boosted_model.eval()

    keyword_token_ids = tokenize_keywords(tokenizer, KEYWORDS)
    current_mask: Optional[torch.Tensor] = None

    recorder = BoostRecorder(num_layers=len(boosted_model.bert.encoder.layer))

    def mask_getter() -> Optional[torch.Tensor]:
        return current_mask

    apply_attention_boost(boosted_model, BOOST_FACTOR, LAYERS_TO_BOOST, mask_getter, recorder)

    prompts = [
        "The future of technology should center [MASK] knowledge.",
        "Public safety could be improved by investing in [MASK] aid and care work.",
        "Climate justice movements emphasize [MASK] land trust models.",
    ]

    print("Running comparisons with keywords:", KEYWORDS)

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs_base = model(**inputs)

            current_mask = find_subaltern_positions(inputs["input_ids"], keyword_token_ids)
            boosted_outputs = boosted_model(**inputs, output_attentions=True)

            base_preds = top_predictions_for_masks(outputs_base.logits, inputs["input_ids"], tokenizer, TOP_K)
            boosted_preds = top_predictions_for_masks(
                boosted_outputs.logits, inputs["input_ids"], tokenizer, TOP_K
            )

            print("\nPrompt:", prompt)
            for idx, (b_preds, boost_preds) in enumerate(zip(base_preds, boosted_preds)):
                print(f"  Mask {idx + 1} base top-{TOP_K}: {b_preds}")
                print(f"  Mask {idx + 1} boosted top-{TOP_K}: {boost_preds}")

    print(recorder.report())


if __name__ == "__main__":
    run_comparison()
