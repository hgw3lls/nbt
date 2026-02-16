"""Adapter shims for console voice nodes with CI-safe mock defaults."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class LLMToken:
    text: str
    logprob_proxy: float


class LLMAdapter(Protocol):
    def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
    ) -> list[LLMToken]: ...


class DiffusionAdapter(Protocol):
    def generate_image(
        self,
        *,
        prompt: str,
        guidance_scale: float,
        embedding: list[float] | None,
        output_path: Path,
    ) -> dict[str, Any]: ...


class MockLLMAdapter:
    """Deterministic token stream for tests and CI."""

    def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
    ) -> list[LLMToken]:
        del top_k
        seed_words = (prompt.strip() or "neural bending").split()
        tokens = []
        for idx in range(max_new_tokens):
            word = seed_words[idx % len(seed_words)]
            entropy = min(1.0, abs(temperature - top_p) + 0.05 * (idx % 5))
            tokens.append(LLMToken(text=f"{word} ", logprob_proxy=-entropy))
        return tokens


class HFCausalLLMAdapterWrapper:
    """Wrapper around existing HF adapter with pseudo-streaming."""

    def __init__(self, model_name: str = "distilgpt2", device: str = "cpu") -> None:
        from neural_bending_toolkit.models.llm_hf import HuggingFaceCausalLMAdapter

        self._adapter = HuggingFaceCausalLMAdapter(model_name=model_name, device=device)

    def stream_generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
    ) -> list[LLMToken]:
        result = self._adapter.generate(
            prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )
        words = result.text.split() or [""]
        logprobs = result.token_logprobs or []
        tokens: list[LLMToken] = []
        for idx, word in enumerate(words):
            lp = float(logprobs[idx]) if idx < len(logprobs) else -0.5
            tokens.append(LLMToken(text=f"{word} ", logprob_proxy=lp))
        return tokens


class MockDiffusionAdapter:
    """Writes placeholder image metadata file for deterministic tests."""

    def generate_image(
        self,
        *,
        prompt: str,
        guidance_scale: float,
        embedding: list[float] | None,
        output_path: Path,
    ) -> dict[str, Any]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prompt": prompt,
            "guidance_scale": guidance_scale,
            "embedding_dim": len(embedding or []),
        }
        output_path.write_text(str(payload), encoding="utf-8")
        return {"image_path": str(output_path), "step_entropy_proxy": 0.2 + 0.01 * len(prompt)}


class DiffusersAdapterWrapper:
    """Wrapper using existing diffusers adapter and artifact saving."""

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", device: str = "cpu") -> None:
        from neural_bending_toolkit.models.diffusion_diffusers import DiffusersStableDiffusionAdapter

        self._adapter = DiffusersStableDiffusionAdapter(model_id=model_id, device=device)

    def generate_image(
        self,
        *,
        prompt: str,
        guidance_scale: float,
        embedding: list[float] | None,
        output_path: Path,
    ) -> dict[str, Any]:
        del embedding
        result = self._adapter.generate(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=20,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if result.images:
            result.images[0].save(output_path)
        return {
            "image_path": str(output_path),
            "step_entropy_proxy": 0.5,
            "attention_maps": len(result.attention_heatmaps),
        }


def _mock_enabled() -> bool:
    return os.getenv("NBT_CONSOLE_USE_MOCK_ADAPTERS", "true").lower() in {
        "1",
        "true",
        "yes",
    }


def adapter_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]


def make_llm_adapter(params: dict[str, Any]) -> LLMAdapter:
    if _mock_enabled():
        return MockLLMAdapter()
    return HFCausalLLMAdapterWrapper(
        model_name=str(params.get("model_name", "distilgpt2")),
        device=str(params.get("device", "cpu")),
    )


def make_diffusion_adapter(params: dict[str, Any]) -> DiffusionAdapter:
    if _mock_enabled():
        return MockDiffusionAdapter()
    return DiffusersAdapterWrapper(
        model_id=str(params.get("model_id", "runwayml/stable-diffusion-v1-5")),
        device=str(params.get("device", "cpu")),
    )
