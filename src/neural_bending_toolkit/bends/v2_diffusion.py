"""Diffusion cross-attention compilation for Bend v2 plans."""

from __future__ import annotations

import math
import re
from typing import Any, Protocol

from neural_bending_toolkit.bends.v2 import (
    BendPlan,
    BendSpec,
    bend_localizability_label,
)


class _MetricTracer(Protocol):
    def log(
        self,
        *,
        step: int,
        metric_name: str,
        value: float | int,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...


def _layer_matches(bend: BendSpec, layer_name: str) -> bool:
    site = bend.site
    if site.kind != "diffusion.cross_attention":
        return False
    if site.allow_all_layers:
        return True
    if site.layer_names and layer_name not in site.layer_names:
        return False
    if site.layer_regex and re.search(site.layer_regex, layer_name) is None:
        return False
    return bool(site.layer_names or site.layer_regex)


def _step_matches(bend: BendSpec, step: int) -> bool:
    site = bend.site
    if site.timestep_start is not None and step < site.timestep_start:
        return False
    if site.timestep_end is not None and step > site.timestep_end:
        return False
    return True


def _schedule_strength(bend: BendSpec, step: int) -> float:
    schedule = bend.schedule
    base = schedule.strength

    if schedule.mode in {"constant", "window"}:
        return base

    if schedule.mode == "ramp":
        start = bend.site.timestep_start or 0
        if bend.site.timestep_end is not None:
            span = max(1, bend.site.timestep_end - start)
        else:
            span = max(1, int(bend.actuator.params.get("ramp_steps", 100)))
        t = min(max((step - start) / span, 0.0), 1.0)
        start_value = (
            schedule.strength_start if schedule.strength_start is not None else 0.0
        )
        end_value = schedule.strength_end if schedule.strength_end is not None else base
        return start_value + (end_value - start_value) * t

    if schedule.mode == "pulse":
        if schedule.period is None or schedule.duty is None:
            return 0.0
        position = step % schedule.period
        active = position < schedule.period * schedule.duty
        return base if active else 0.0

    return 0.0


def _infer_num_heads(tensor: Any, hint: int | None = None) -> int:
    if hint is not None and hint > 0 and tensor.shape[0] % hint == 0:
        return hint
    return int(tensor.shape[0])


def _head_batch_view(tensor: Any, num_heads: int) -> Any:
    batch = tensor.shape[0] // num_heads
    return tensor.reshape(batch, num_heads, *tensor.shape[1:])


def _selected_heads(head_count: int, head_indices: list[int] | None) -> list[int]:
    if head_indices is None:
        return list(range(head_count))
    return [idx for idx in head_indices if 0 <= idx < head_count]


def _rotate_qk(query: Any, key: Any, heads: list[int], angle: float) -> tuple[Any, Any]:
    if not heads or query.shape[-1] < 2:
        return query, key

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    def _apply_rotation(tensor: Any) -> Any:
        target = tensor[:, heads]
        first = target[..., 0].clone()
        second = target[..., 1].clone()
        target[..., 0] = cos_a * first - sin_a * second
        target[..., 1] = sin_a * first + cos_a * second
        tensor[:, heads] = target
        return tensor

    return _apply_rotation(query), _apply_rotation(key)


def _log_attention_entropy(
    *,
    tracer: _MetricTracer | None,
    attention_probs: Any,
    step: int,
    bend_name: str,
    layer_name: str,
    sample_every: int,
    localizability: str,
) -> None:
    if tracer is None or sample_every <= 0 or step % sample_every != 0:
        return

    probs = attention_probs.clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
    tracer.log(
        step=step,
        metric_name="attention_entropy",
        value=float(entropy),
        metadata={
            "bend": bend_name,
            "layer": layer_name,
            "localizability": localizability,
        },
    )


def compile_diffusion_cross_attention_hook(
    plan: BendPlan,
    tracer: _MetricTracer | None = None,
) -> Any:
    """Compile a BendPlan into a diffusers-compatible cross-attention hook."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to compile diffusion bend hooks") from exc

    diffusion_bends = [
        bend for bend in plan.bends if bend.site.kind == "diffusion.cross_attention"
    ]

    if not diffusion_bends:
        return lambda _payload: None

    def _hook(payload: dict[str, Any]) -> dict[str, Any] | None:
        layer_name = str(payload.get("layer", ""))
        step = int(payload.get("step", payload.get("timestep", 0)))
        query = payload.get("query")
        key = payload.get("key")
        value = payload.get("value")
        attention_probs = payload.get("attention_probs")

        if query is None or key is None or attention_probs is None:
            return None

        modified = False
        for bend in diffusion_bends:
            if not _layer_matches(bend, layer_name) or not _step_matches(bend, step):
                continue

            strength = _schedule_strength(bend, step)
            if strength == 0:
                continue

            num_heads_hint = bend.actuator.params.get("num_heads")
            num_heads = _infer_num_heads(attention_probs, num_heads_hint)
            if attention_probs.shape[0] % num_heads != 0:
                continue

            attn_heads = _head_batch_view(attention_probs, num_heads)
            q_heads = _head_batch_view(query, num_heads)
            k_heads = _head_batch_view(key, num_heads)
            v_heads = _head_batch_view(value, num_heads) if value is not None else None

            heads = _selected_heads(num_heads, bend.site.head_indices)
            if not heads:
                continue

            if bend.actuator.type == "attention_head_gate":
                scale = float(bend.actuator.params.get("scale", 0.0))
                effective_scale = 1.0 - strength * (1.0 - scale)
                attn_heads[:, heads] = attn_heads[:, heads] * effective_scale
                modified = True

            elif bend.actuator.type == "attention_probs_temperature":
                default_temp = max(strength, 1e-6)
                temperature = float(
                    bend.actuator.params.get("temperature", default_temp)
                )
                logits = (attn_heads[:, heads].clamp_min(1e-12)).log() / max(
                    temperature,
                    1e-6,
                )
                attn_heads[:, heads] = torch.softmax(logits, dim=-1)
                modified = True

            elif bend.actuator.type == "qk_rotate":
                angle = float(
                    bend.actuator.params.get(
                        "angle",
                        bend.actuator.params.get("eps", 0.05 * strength),
                    )
                )
                q_heads, k_heads = _rotate_qk(q_heads, k_heads, heads, angle)
                modified = True

            elif bend.actuator.type == "kv_noise" and v_heads is not None:
                sigma = float(bend.actuator.params.get("sigma", 0.01 * strength))
                k_heads[:, heads] = k_heads[:, heads] + torch.randn_like(
                    k_heads[:, heads]
                ) * sigma
                v_heads[:, heads] = v_heads[:, heads] + torch.randn_like(
                    v_heads[:, heads]
                ) * sigma
                modified = True

            if bend.trace and "attention_entropy" in bend.trace.metrics:
                _log_attention_entropy(
                    tracer=tracer,
                    attention_probs=attn_heads,
                    step=step,
                    bend_name=bend.name,
                    layer_name=layer_name,
                    sample_every=bend.trace.sample_every,
                    localizability=bend_localizability_label(bend),
                )

            attention_probs = attn_heads.reshape_as(attention_probs)
            query = q_heads.reshape_as(query)
            key = k_heads.reshape_as(key)
            if value is not None and v_heads is not None:
                value = v_heads.reshape_as(value)

        if not modified:
            return None

        return {
            "query": query,
            "key": key,
            "value": value,
            "attention_probs": attention_probs,
        }

    return _hook
