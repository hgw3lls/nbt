"""Diffusers adapter for Stable Diffusion-style pipelines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np

from neural_bending_toolkit.models.torch_device import normalize_torch_device
from neural_bending_toolkit.models.hooks import register_forward_hook, register_forward_pre_hook

logger = logging.getLogger(__name__)


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
NormHook = Callable[[dict[str, Any]], Any]



class _NormHookManager:
    """Manage lifecycle and safety behavior for normalization hooks."""

    def __init__(self, hook: NormHook, get_step: Callable[[], int]) -> None:
        self._hook = hook
        self._get_step = get_step
        self._handles: list[Any] = []
        self._enabled = True

    def register_module(self, layer_name: str, module: Any) -> None:
        handle = register_forward_hook(module, self._make_post_hook(layer_name))
        self._handles.append(handle)

    def remove_all(self) -> None:
        while self._handles:
            self._handles.pop().remove()

    def _disable_with_warning(self, exc: Exception, layer_name: str) -> None:
        if self._enabled:
            logger.warning("Disabling normalization hooks after error in %s: %s", layer_name, exc)
            self._enabled = False
            self.remove_all()

    def _make_post_hook(self, layer_name: str) -> Callable[..., Any]:
        def _post_hook(module: Any, args: tuple[Any, ...], output: Any) -> Any:
            if not self._enabled:
                return output
            payload = {"layer": layer_name, "step": self._get_step(), "tensor": output, "module": module, "metadata": {"hook_kind": "post", "input_count": len(args)}}
            try:
                updated = self._hook(payload)
            except TypeError:
                try:
                    from types import SimpleNamespace
                    updated = self._hook(output, SimpleNamespace(layer_name=layer_name, step=payload["step"], metadata=payload["metadata"], cache=payload.get("cache", {})))  # type: ignore[misc]
                except Exception as exc:
                    self._disable_with_warning(exc, layer_name)
                    return output
            except Exception as exc:
                self._disable_with_warning(exc, layer_name)
                return output
            return output if updated is None else updated

        return _post_hook


class _ResidualHookManager:

    """Lifecycle management for residual stream forward hooks."""

    def __init__(
        self,
        *,
        hook: ResidualHook,
        get_step: Callable[[], int],
        cache: dict[str, dict[str, Any]],
        max_layers_cached: int | None,
        store_fp16_on_cuda: bool,
        fp16_dtype: Any,
    ) -> None:
        self._hook = hook
        self._get_step = get_step
        self._cache = cache
        self._max_layers_cached = max_layers_cached
        self._store_fp16_on_cuda = store_fp16_on_cuda
        self._fp16_dtype = fp16_dtype
        self._handles: list[Any] = []
        self._enabled = True

    def register_module(self, layer_name: str, module: Any) -> None:
        handle = register_forward_hook(module, self._make_post_hook(layer_name))
        self._handles.append(handle)

    def remove_all(self) -> None:
        while self._handles:
            handle = self._handles.pop()
            handle.remove()

    def _disable_with_warning(self, exc: Exception, layer_name: str) -> None:
        if self._enabled:
            logger.warning(
                "Disabling residual hooks after error in %s: %s",
                layer_name,
                exc,
            )
            self._enabled = False
            self.remove_all()

    def _bounded_cache(self, layer_name: str, value: Any) -> None:
        echo_cache = self._cache["residual_echo"]
        if layer_name not in echo_cache:
            if self._max_layers_cached is not None:
                if self._max_layers_cached <= 0:
                    return
                if len(echo_cache) >= self._max_layers_cached:
                    return

        cached = value.detach()
        if (
            self._store_fp16_on_cuda
            and getattr(cached, "is_cuda", False)
            and getattr(cached, "dtype", None) is not None
            and cached.dtype != self._fp16_dtype
        ):
            cached = cached.to(dtype=self._fp16_dtype)
        echo_cache[layer_name] = cached

    def _make_post_hook(self, layer_name: str) -> Callable[..., Any]:
        def _post_hook(module: Any, args: tuple[Any, ...], output: Any) -> Any:
            if not self._enabled:
                return output
            prev = self._cache["residual_echo"].get(layer_name)
            payload = {
                "layer": layer_name,
                "step": self._get_step(),
                "tensor": output,
                "prev": prev,
                "cache": self._cache,
                "module": module,
                "metadata": {"hook_kind": "post", "input_count": len(args)},
            }
            try:
                updated = self._hook(payload)
            except TypeError:
                try:
                    from types import SimpleNamespace
                    updated = self._hook(output, SimpleNamespace(layer_name=layer_name, step=payload["step"], metadata=payload["metadata"], cache=payload.get("cache", {})))  # type: ignore[misc]
                except Exception as exc:
                    self._disable_with_warning(exc, layer_name)
                    return output
            except Exception as exc:  # pragma: no cover - behavior validated elsewhere
                self._disable_with_warning(exc, layer_name)
                return output

            final_output = output if updated is None else updated
            self._bounded_cache(layer_name, final_output)
            return final_output

        return _post_hook


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
                "value": value,
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
                value = modified.get("value", value)
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

    def _iter_normalization_modules(self) -> list[tuple[str, Any]]:
        torch = self._torch
        norm_types = (torch.nn.GroupNorm, torch.nn.LayerNorm)
        modules: list[tuple[str, Any]] = []
        for name, module in self._pipe.unet.named_modules():
            if not name:
                continue
            if isinstance(module, norm_types) or "LayerNorm" in type(module).__name__:
                modules.append((name, module))
        return modules

    def _iter_residual_modules(self, layer_pattern: str) -> list[tuple[str, Any]]:
        pattern = re.compile(layer_pattern)
        return [
            (name, module)
            for name, module in self._pipe.unet.named_modules()
            if name and pattern.search(name)
        ]

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
        norm_hook: NormHook | None = None,
        residual_hook: ResidualHook | None = None,
        residual_layer_pattern: str = r"(?:resnets|transformer_blocks)",
        max_cache_layers: int = 64,
        max_layers_cached: int | None = None,
        residual_cache_fp16_on_cuda: bool = True,
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

        self.current_step = 0
        residual_cache: dict[str, dict[str, Any]] = {"residual_echo": {}}
        norm_manager: _NormHookManager | None = None
        if norm_hook is not None:
            norm_manager = _NormHookManager(
                hook=norm_hook,
                get_step=lambda: self.current_step,
            )
            for layer_name, module in self._iter_normalization_modules():
                norm_manager.register_module(layer_name, module)

        residual_manager: _ResidualHookManager | None = None
        if residual_hook is not None:
            residual_manager = _ResidualHookManager(
                hook=residual_hook,
                get_step=lambda: self.current_step,
                cache=residual_cache,
                max_layers_cached=max_layers_cached if max_layers_cached is not None else max_cache_layers,
                store_fp16_on_cuda=residual_cache_fp16_on_cuda,
                fp16_dtype=torch.float16,
            )
            for layer_name, module in self._iter_residual_modules(residual_layer_pattern):
                residual_manager.register_module(layer_name, module)

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

        try:
            result = self._pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=_step_callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        finally:
            if norm_manager is not None:
                norm_manager.remove_all()
            if residual_manager is not None:
                residual_manager.remove_all()

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
