"""Attention bends that bias relevance toward neglected tokens."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from .base import BendResult, NeuralBend, register_bend


def _parse_tokens(tokenizer: Any, token_csv: str) -> Tuple[List[str], List[int]]:
    tokens = [tok.strip() for tok in token_csv.split(",") if tok.strip()]
    ids: List[int] = []
    for tok in tokens:
        encoding = tokenizer.encode(tok, add_special_tokens=False)
        if encoding:
            ids.append(encoding[0])
    return tokens, ids


class BendingLogitsProcessor(LogitsProcessor):
    """Patch-point that lets us over-amplify specific vocab items."""

    def __init__(
        self,
        tokenizer: Any,
        boost_ids: Iterable[int],
        suppress_ids: Iterable[int],
        boost_scale: float,
        suppress_scale: float,
        mode: str,
        mix: float,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.boost_ids = list(boost_ids)
        self.suppress_ids = list(suppress_ids)
        self.boost_scale = boost_scale
        self.suppress_scale = suppress_scale
        self.mode = mode
        self.mix = mix
        self.step_logs: List[dict] = []
        self.gain_counter: Counter[str] = Counter()
        self.device = torch.device("cpu")
        self.vocab_size = 0

    def _make_delta(self, ids_to_boost: Iterable[int], ids_to_suppress: Iterable[int]) -> torch.Tensor:
        delta = torch.zeros(1, self.vocab_size, device=self.device)
        if ids_to_boost:
            delta[:, ids_to_boost] += self.boost_scale
        if ids_to_suppress:
            delta[:, ids_to_suppress] += self.suppress_scale
        return delta

    def _log_step(self, ids_boosted: Iterable[int], ids_suppressed: Iterable[int], stream: str) -> None:
        boosted_tokens = [self.tokenizer.convert_ids_to_tokens(i) for i in ids_boosted]
        suppressed_tokens = [self.tokenizer.convert_ids_to_tokens(i) for i in ids_suppressed]
        for tok in boosted_tokens:
            self.gain_counter[tok] += 1
        self.step_logs.append(
            {
                "step": len(self.step_logs),
                "stream": stream,
                "boosted": boosted_tokens,
                "suppressed": suppressed_tokens,
            }
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:  # type: ignore[override]
        self.device = scores.device
        self.vocab_size = scores.shape[-1]
        base_scores = scores

        if self.mode == "boost":
            delta = self._make_delta(self.boost_ids, self.suppress_ids)
            self._log_step(self.boost_ids, self.suppress_ids, stream="boost")
            bent_scores = base_scores + delta
        elif self.mode == "divergent":
            delta_a = self._make_delta(self.boost_ids, self.suppress_ids)
            delta_b = self._make_delta(self.suppress_ids, self.boost_ids)
            self._log_step(self.boost_ids, self.suppress_ids, stream="A")
            self._log_step(self.suppress_ids, self.boost_ids, stream="B")
            bent_scores = base_scores + 0.5 * (delta_a + delta_b)
        else:
            bent_scores = base_scores

        # Dry/wet mix to emulate patching into the relevance bus
        mixed_scores = (1.0 - self.mix) * base_scores + self.mix * bent_scores
        return mixed_scores


class MinorityAttentionReweight(NeuralBend):
    """Tilt attention toward minoritized token clusters.

    Technical: uses a logits processor to boost chosen tokens and dampen
    dominant ones, preserving baseline flow via a dry/wet mix.
    Media-theoretical: exposes how attention routes normalize visibility; the
    bend reclaims relevance budget for counter-hegemonic fragments.
    """

    def __init__(self) -> None:
        super().__init__(
            name="attention_minority_reweight",
            domain="attention",
            category="revelatory",
            description="Bias attention toward minority tokens while tracking who gets amplified.",
            technical_notes="Return a Hugging Face LogitsProcessorList to feed into generate().",
        )

    def apply(
        self,
        model: Any,
        *,
        tokenizer: Any,
        boost_tokens: str,
        suppress_tokens: str,
        boost_scale: float = 5.0,
        suppress_scale: float = -5.0,
        mode: str = "boost",
        mix: float = 0.5,
    ) -> BendResult:
        if mode not in {"boost", "divergent"}:
            raise ValueError("mode must be 'boost' or 'divergent'")
        boost_list, boost_ids = _parse_tokens(tokenizer, boost_tokens)
        suppress_list, suppress_ids = _parse_tokens(tokenizer, suppress_tokens)

        processor = BendingLogitsProcessor(
            tokenizer=tokenizer,
            boost_ids=boost_ids,
            suppress_ids=suppress_ids,
            boost_scale=boost_scale,
            suppress_scale=suppress_scale,
            mode=mode,
            mix=mix,
        )
        logits_processors = LogitsProcessorList([processor])

        metadata: Dict[str, Any] = {
            "processor": logits_processors,
            "boost_tokens": boost_list,
            "suppress_tokens": suppress_list,
            "gain_map": processor.gain_counter,
        }
        return BendResult(model=model, metadata=metadata)


register_bend(MinorityAttentionReweight())
