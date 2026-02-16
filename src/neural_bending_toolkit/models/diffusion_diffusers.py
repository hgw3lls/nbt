"""Diffusers adapter for Stable Diffusion-style pipelines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neural_bending_toolkit.models.torch_device import normalize_torch_device


class _TorchXPUStub:
    """Fallback `torch.xpu` namespace for environments without Intel XPU support."""

    @staticmethod
    def empty_cache() -> None:
        """No-op empty cache implementation used by diffusers import-time checks."""

    @staticmethod
    def is_available() -> bool:
        """Report unavailable XPU support in CPU-only environments."""

        return False


def _ensure_torch_xpu_namespace(torch_module: Any) -> None:
    """Provide `torch.xpu` when absent so diffusers can import on CPU-only builds."""

    if not hasattr(torch_module, "xpu"):
        torch_module.xpu = _TorchXPUStub()  # type: ignore[attr-defined]


@dataclass
class DiffusionGenerationResult:
    """Container for generated images and captured attention artifacts."""

    images: list[Any]
    attention_heatmaps: dict[str, list[np.ndarray]]


EmbeddingHook = Callable[[Any, dict[str, Any]], Any]
CrossAttentionHook = Callable[[dict[str, Any]], dict[str, Any] | None]


class HookedCrossAttentionProcessor:
    """Cross-attention processor that supports interception and modification."""

    def __init__(
        self,
        layer_name: str,
        get_step: Callable[[], int],
        hook: CrossAttentionHook | None,
        heatmaps: dict[str, list[np.ndarray]],
    ) -> None:
        self.layer_name = layer_name
        self.get_step = get_step
        self.hook = hook
        self.heatmaps = heatmaps

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any = None,
        attention_mask: Any = None,
        temb: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        del args, kwargs
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size,
                channel,
                height * width,
            ).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask,
            sequence_length,
            batch_size,
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2))
            hidden_states = hidden_states.transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states,
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        step_idx = self.get_step()
        heatmap = attention_probs.detach().float().mean(dim=0).cpu().numpy()
        bucket = self.heatmaps.setdefault(f"{self.layer_name}/step_{step_idx}", [])
        bucket.append(heatmap)

        if self.hook is not None:
            payload = {
                "layer": self.layer_name,
                "step": step_idx,
                "query": query,
                "key": key,
                "attention_probs": attention_probs,
            }
            modified = self.hook(payload)
            if modified is not None:
                if not isinstance(modified, dict):
                    raise TypeError(
                        "cross_attention_hook must return a dict[str, Any] or None",
                    )
                query = modified.get("query", query)
                key = modified.get("key", key)
                attention_probs = modified.get("attention_probs", attention_probs)

        hidden_states = attn.batch_to_head_dim(attention_probs @ value)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size,
                channel,
                height,
                width,
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class DiffusersStableDiffusionAdapter:
    """Adapter around diffusers pipelines with embedding and attention hooks."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        torch_dtype: str = "float32",
    ) -> None:
        try:
            import torch

            _ensure_torch_xpu_namespace(torch)
            from diffusers import StableDiffusionPipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "diffusers + torch are required for DiffusersStableDiffusionAdapter. "
                "Install with: pip install .[diffusion]"
            ) from exc
        except AttributeError as exc:
            raise RuntimeError(
                "Incompatible torch/diffusers installation. "
                "Try upgrading torch and diffusers together, "
                "or reinstall with: pip install --upgrade '.[diffusion]'"
            ) from exc

        self._torch = torch
        self.device = normalize_torch_device(device)
        self._pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, torch_dtype),
        )
        self._pipe = self._pipe.to(self.device)
        self.current_step = 0

    def _encode_prompt(self, prompt: str) -> Any:
        torch = self._torch

        tokenized = self._pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self._pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(self.device)
        with torch.no_grad():
            return self._pipe.text_encoder(input_ids)[0]

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        embedding_hook: EmbeddingHook | None = None,
        cross_attention_hook: CrossAttentionHook | None = None,
        generator_seed: int | None = None,
    ) -> DiffusionGenerationResult:
        torch = self._torch

        prompt_embeds = self._encode_prompt(prompt)
        hook_context = {"prompt": prompt}
        if embedding_hook is not None:
            prompt_embeds = embedding_hook(prompt_embeds, hook_context)

        heatmaps: dict[str, list[np.ndarray]] = {}
        processor_map: dict[str, Any] = {}
        for layer_name in self._pipe.unet.attn_processors.keys():
            processor_map[layer_name] = HookedCrossAttentionProcessor(
                layer_name=layer_name,
                get_step=lambda: self.current_step,
                hook=cross_attention_hook,
                heatmaps=heatmaps,
            )
        self._pipe.unet.set_attn_processor(processor_map)

        generator = None
        if generator_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(generator_seed)

        def _step_callback(
            _: Any,
            step: int,
            timestep: Any,
            callback_kwargs: dict[str, Any],
        ) -> dict[str, Any]:
            del timestep
            self.current_step = step
            return callback_kwargs

        result = self._pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=_step_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        return DiffusionGenerationResult(
            images=result.images,
            attention_heatmaps=heatmaps,
        )

    def save_artifacts(
        self,
        output: DiffusionGenerationResult,
        artifacts_dir: Path,
        *,
        prefix: str,
    ) -> list[Path]:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []

        for idx, image in enumerate(output.images):
            path = artifacts_dir / f"{prefix}_image_{idx}.png"
            image.save(path)
            saved_paths.append(path)

        for key, maps in output.attention_heatmaps.items():
            safe_key = key.replace("/", "_")
            for idx, heatmap in enumerate(maps):
                path = artifacts_dir / f"{prefix}_{safe_key}_{idx}.npy"
                np.save(path, heatmap)
                saved_paths.append(path)

        return saved_paths
