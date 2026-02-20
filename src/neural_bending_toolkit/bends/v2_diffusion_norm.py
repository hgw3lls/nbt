"""Diffusion normalization-hook compilation for Bend v2 plans."""

from __future__ import annotations

import re
from typing import Any, Protocol

from neural_bending_toolkit.bends.v2 import BendPlan, BendSpec, bend_localizability_label


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
    if schedule.mode in {"constant", "window"}:
        return schedule.strength
    if schedule.mode == "ramp":
        start = bend.site.timestep_start or 0
        span = max(1, (bend.site.timestep_end or (start + 100)) - start)
        t = min(max((step - start) / span, 0.0), 1.0)
        start_value = schedule.strength_start if schedule.strength_start is not None else 0.0
        end_value = schedule.strength_end if schedule.strength_end is not None else schedule.strength
        return start_value + (end_value - start_value) * t
    if schedule.mode == "pulse":
        if schedule.period is None or schedule.duty is None:
            return 0.0
        return schedule.strength if (step % schedule.period) < schedule.period * schedule.duty else 0.0
    return 0.0


def compile_diffusion_norm_hook(plan: BendPlan, tracer: _MetricTracer | None = None) -> Any:
    """Compile a BendPlan into a diffusers norm hook: hook(payload)->tensor|None."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to compile diffusion norm hooks") from exc

    bends = [bend for bend in plan.bends if bend.site.kind == "diffusion.norm"]
    if not bends:
        return lambda _payload: None

    def _hook(payload: dict[str, Any]) -> Any | None:
        layer_name = str(payload.get("layer", ""))
        step = int(payload.get("step", payload.get("timestep", 0)))
        x = payload.get("tensor")
        if x is None:
            return None

        current = x
        modified = False
        for bend in bends:
            if not _layer_matches(bend, layer_name) or not _step_matches(bend, step):
                continue
            strength = _schedule_strength(bend, step)
            if strength == 0:
                continue

            params = bend.actuator.params
            if bend.actuator.type == "norm_gain_drift":
                scale = float(params.get("scale", 1.0))
                current = current * (1.0 + strength * scale)
                modified = True
            elif bend.actuator.type == "norm_bias_shift":
                bias = float(params.get("bias", 1.0))
                current = current + strength * bias
                modified = True
            elif bend.actuator.type == "activation_noise":
                sigma = float(params.get("sigma", 0.01 * abs(strength)))
                current = current + torch.randn_like(current) * sigma
                modified = True

            trace = bend.trace
            if tracer is None or trace is None or step % trace.sample_every != 0:
                continue
            metadata = {
                "bend": bend.name,
                "layer": layer_name,
                "localizability": bend_localizability_label(bend),
            }
            if "norm_output_mean" in trace.metrics:
                tracer.log(step=step, metric_name="norm_output_mean", value=float(current.mean().item()), metadata=metadata)
            if "norm_output_var" in trace.metrics:
                tracer.log(step=step, metric_name="norm_output_var", value=float(current.var(unbiased=False).item()), metadata=metadata)

        return current if modified else None

    return _hook
