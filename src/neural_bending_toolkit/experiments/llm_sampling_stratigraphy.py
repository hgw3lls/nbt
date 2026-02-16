"""Sampling stratigraphy experiment for causal language models."""

from __future__ import annotations

import itertools
import json

import numpy as np
from pydantic import Field

from neural_bending_toolkit.experiment import Experiment, ExperimentSettings, RunContext


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Compute KL(P || Q) for probability vectors."""
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))


class LLMSamplingStratigraphyConfig(ExperimentSettings):
    """Config for LLM sampling stratigraphy."""

    model_name: str = "sshleifer/tiny-gpt2"
    prompt: str = "The neural manifold is"
    max_new_tokens: int = Field(default=24, ge=1, le=256)
    temperatures: list[float] = Field(default_factory=lambda: [0.7, 1.0, 1.3])
    top_ps: list[float] = Field(default_factory=lambda: [0.85, 0.95, 1.0])
    top_ks: list[int] = Field(default_factory=lambda: [0, 20, 50])
    baseline_temperature: float = 1.0
    baseline_top_p: float = 1.0
    baseline_top_k: int = 0
    analyze_hidden_layers: list[int] = Field(default_factory=lambda: [0])
    analyze_attention_layers: list[int] = Field(default_factory=lambda: [0])


class LLMSamplingStratigraphy(Experiment):
    """Vary temperature/top-p/top-k and compare distributions to a baseline."""

    name = "llm-sampling-stratigraphy"
    config_model = LLMSamplingStratigraphyConfig

    def _load_adapter(self):
        from neural_bending_toolkit.models.llm_hf import HuggingFaceCausalLMAdapter

        return HuggingFaceCausalLMAdapter(self.config.model_name, device="cpu")

    def run(self, context: RunContext) -> None:
        adapter = self._load_adapter()

        baseline_dist = adapter.sampling_distribution(
            self.config.prompt,
            temperature=self.config.baseline_temperature,
            top_p=self.config.baseline_top_p,
            top_k=self.config.baseline_top_k,
        )

        signals = adapter.capture_model_signals(
            self.config.prompt,
            hidden_layers=self.config.analyze_hidden_layers,
            attention_layers=self.config.analyze_attention_layers,
            logit_positions=[-1],
        )
        context.save_text_artifact(
            "signal_capture.json",
            json.dumps(signals, indent=2),
        )

        step = 0
        for temperature, top_p, top_k in itertools.product(
            self.config.temperatures,
            self.config.top_ps,
            self.config.top_ks,
        ):
            step += 1
            context.pre_intervention_snapshot(
                name="sampling_params",
                data={"temperature": temperature, "top_p": top_p, "top_k": top_k},
            )

            generation = adapter.generate(
                self.config.prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
            )

            current_dist = adapter.sampling_distribution(
                self.config.prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            kl = kl_divergence(current_dist, baseline_dist)
            first_logprob = None
            if generation.token_logprobs:
                first_logprob = generation.token_logprobs[0]

            context.log_metric(
                step=step,
                metric_name="kl_vs_baseline",
                value=kl,
                metadata={
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "first_token_logprob": first_logprob,
                },
            )
            context.log_event(
                "Generated candidate",
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                text=generation.text,
            )
            context.post_intervention_snapshot(
                name="sampling_outcome",
                data={"kl_vs_baseline": kl, "text": generation.text[:200]},
            )

            artifact_name = f"samples/t{temperature:.2f}_p{top_p:.2f}_k{top_k}.txt"
            context.save_text_artifact(artifact_name, generation.text)
