"""Diffusion normalization-hook compilation for Bend v2 plans."""

from __future__ import annotations

import math
import re
from typing import Any, Protocol

from neural_bending_toolkit.bends.v2 import BendPlan, BendSpec, bend_localizability_label
from neural_bending_toolkit.models.hooks import HookContext


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
    if site.kind != "diffusion.norm":
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


def _channel_mask_like(x: Any, mask_values: list[float] | None) -> Any:
    if not mask_values:
        return x.new_ones(x.shape)

    channel_dim = 1 if x.ndim >= 2 else 0
    channels = x.shape[channel_dim]
    if channels <= 0:
        return x.new_ones(x.shape)

    values = list(mask_values)
    if len(values) < channels:
        values.extend([1.0] * (channels - len(values)))
    if len(values) > channels:
        values = values[:channels]

    mask = x.new_tensor(values)
    view_shape = [1] * x.ndim
    view_shape[channel_dim] = channels
    return mask.view(*view_shape)


def _apply_norm_stat_clamp(x: Any, strength: float, params: dict[str, Any]) -> Any:
    eps = float(params.get("eps", 1e-6))
    min_var = params.get("min_var")
    max_var = params.get("max_var")

    if x.ndim <= 1:
        return x

    dims = tuple(range(1, x.ndim))
    mean = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, keepdim=True, unbiased=False)
    normalized = (x - mean) / (var + eps).sqrt()

    min_std = math.sqrt(float(min_var)) if min_var is not None else None
    max_std = math.sqrt(float(max_var)) if max_var is not None else None
    std = (var + eps).sqrt()
    if min_std is not None:
        std = std.clamp_min(min_std)
    if max_std is not None:
        std = std.clamp_max(max_std)

    perturbed = normalized * std + mean
    return x + strength * (perturbed - x)


def _log_metrics(
    *,
    tracer: _MetricTracer | None,
    bend: BendSpec,
    step: int,
    layer_name: str,
    x: Any,
) -> None:
    trace = bend.trace
    if tracer is None or trace is None or step % trace.sample_every != 0:
        return

    metadata = {
        "bend": bend.name,
        "layer": layer_name,
        "localizability": bend_localizability_label(bend),
    }

    if "norm_output_mean" in trace.metrics:
        tracer.log(
            step=step,
            metric_name="norm_output_mean",
            value=float(x.mean().item()),
            metadata=metadata,
        )

    if "norm_output_var" in trace.metrics:
        tracer.log(
            step=step,
            metric_name="norm_output_var",
            value=float(x.var(unbiased=False).item()),
            metadata=metadata,
        )

    if "activation_snr" in trace.metrics:
        eps = float(bend.actuator.params.get("snr_eps", 1e-6))
        snr = x.abs().mean() / (x.std(unbiased=False) + eps)
        tracer.log(
            step=step,
            metric_name="activation_snr",
            value=float(snr.item()),
            metadata=metadata,
        )


def compile_diffusion_norm_hook(
    plan: BendPlan,
    tracer: _MetricTracer | None = None,
) -> Any:
    """Compile a BendPlan into a normalization hook callback."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to compile diffusion norm hooks") from exc

    bends = [bend for bend in plan.bends if bend.site.kind == "diffusion.norm"]
    if not bends:
        return lambda x, _ctx: x

    def _hook(x: Any, ctx: HookContext) -> Any:
        result = x
        for bend in bends:
            if not _layer_matches(bend, ctx.layer_name) or not _step_matches(bend, ctx.step):
                continue

            strength = _schedule_strength(bend, ctx.step)
            if strength == 0:
                continue

            if bend.actuator.type == "norm_gain_drift":
                mask_values = bend.actuator.params.get("channel_mask")
                mask = _channel_mask_like(result, mask_values)
                result = result * (1.0 + strength * mask)
            elif bend.actuator.type == "norm_bias_shift":
                bias = float(bend.actuator.params.get("bias", 1.0))
                result = result + strength * bias
            elif bend.actuator.type == "norm_stat_clamp":
                result = _apply_norm_stat_clamp(result, strength, bend.actuator.params)
            elif bend.actuator.type == "activation_noise":
                sigma = float(bend.actuator.params.get("sigma", 1.0))
                result = result + torch.randn_like(result) * (abs(strength) * sigma)

            _log_metrics(
                tracer=tracer,
                bend=bend,
                step=ctx.step,
                layer_name=ctx.layer_name,
                x=result,
            )

        return result

    return _hook
