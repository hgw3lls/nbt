"""Hugging Face causal language model adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class GenerationResult:
    """Unified generation output."""

    text: str
    token_logprobs: list[float] | None
    token_ids: list[int]


class HuggingFaceCausalLMAdapter:
    """Adapter around Transformers causal LMs for sampling and introspection."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Transformers + torch are required for HuggingFaceCausalLMAdapter. "
                "Install with: pip install .[llm]"
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device

        has_pad = self.tokenizer.pad_token_id is not None
        has_eos = self.tokenizer.eos_token_id is not None
        if not has_pad and has_eos:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, **params: Any) -> GenerationResult:
        """Generate text and token-level logprobs when available."""
        torch = self._torch

        max_new_tokens = int(params.get("max_new_tokens", 32))
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", 0))
        do_sample = bool(params.get("do_sample", True))

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        full_sequence = outputs.sequences[0]
        prompt_len = input_ids.shape[1]
        generated_ids = full_sequence[prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        token_logprobs: list[float] = []
        if outputs.scores:
            for step_idx, score in enumerate(outputs.scores):
                if step_idx >= generated_ids.shape[0]:
                    break
                log_probs = torch.log_softmax(score[0], dim=-1)
                token_id = int(generated_ids[step_idx].item())
                token_logprobs.append(float(log_probs[token_id].item()))

        return GenerationResult(
            text=text,
            token_logprobs=token_logprobs or None,
            token_ids=[int(token.item()) for token in generated_ids],
        )

    def sampling_distribution(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> np.ndarray:
        """Return first-token sampling probability distribution under settings."""
        torch = self._torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits[0, -1, :]
        logits = logits / max(temperature, 1e-8)

        if top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[-1]
            neg_inf = torch.tensor(float("-inf")).to(logits)
            logits = torch.where(logits < kth, neg_inf, logits)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            to_remove = cumulative > top_p
            to_remove[0] = False
            sorted_logits = sorted_logits.masked_fill(to_remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(0, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()

    def capture_model_signals(
        self,
        prompt: str,
        *,
        hidden_layers: list[int] | None = None,
        attention_layers: list[int] | None = None,
        logit_positions: list[int] | None = None,
    ) -> dict[str, Any]:
        """Capture selected hidden states, attention maps, and logits."""
        torch = self._torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
            )

        selected_hidden: dict[str, Any] = {}
        if hidden_layers:
            for layer in hidden_layers:
                hidden = outputs.hidden_states[layer][0].detach().cpu().numpy().tolist()
                selected_hidden[str(layer)] = hidden

        selected_attention: dict[str, Any] = {}
        if attention_layers:
            for layer in attention_layers:
                attn = outputs.attentions[layer][0].detach().cpu().numpy().tolist()
                selected_attention[str(layer)] = attn

        selected_logits: dict[str, Any] = {}
        if logit_positions:
            for pos in logit_positions:
                logits = outputs.logits[0, pos].detach().cpu().numpy().tolist()
                selected_logits[str(pos)] = logits

        return {
            "hidden_states": selected_hidden,
            "attention_maps": selected_attention,
            "logits": selected_logits,
            "tokens": self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
        }
