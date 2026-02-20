"""Diffusion residual-hook compilation for Bend v2 plans."""

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
    if site.kind != "diffusion.residual":
        return False
    if site.allow_all_layers:
        return True
    if site.layer_names and layer_name not in site.layer_names:
        return False
    if site.layer_regex and re.search(site.layer_regex, layer_name) is None:
        return False
    return bool(site.layer_names or site.layer_regex)


def _step_matches(bend: BendSpec, step: int) -> bool:
    if bend.site.timestep_start is not None and step < bend.site.timestep_start:
        return False
    if bend.site.timestep_end is not None and step > bend.site.timestep_end:
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


def compile_diffusion_residual_hook(plan: BendPlan, tracer: _MetricTracer | None = None) -> Any:
    """Compile a BendPlan into a diffusers residual hook: hook(payload)->tensor|None."""

    bends = [bend for bend in plan.bends if bend.site.kind == "diffusion.residual"]
    if not bends:
        return lambda _payload: None

    def _hook(payload: dict[str, Any]) -> Any | None:
        x = payload.get("tensor")
        if x is None:
            return None
        layer_name = str(payload.get("layer", ""))
        step = int(payload.get("step", payload.get("timestep", 0)))
        prev = payload.get("prev")

        current = x
        modified = False
        for bend in bends:
            if not _layer_matches(bend, layer_name) or not _step_matches(bend, step):
                continue
            strength = _schedule_strength(bend, step)
            if strength == 0:
                continue

            before = current
            params = bend.actuator.params
            if bend.actuator.type == "residual_echo" and prev is not None:
                alpha = strength * float(params.get("alpha", 1.0))
                current = current + alpha * prev.to(device=current.device, dtype=current.dtype)
                modified = True
            elif bend.actuator.type == "residual_clamp":
                max_norm = float(params.get("max_norm", 1.0))
                if max_norm > 0:
                    norm = current.norm()
                    if norm.item() > max_norm:
                        current = current * (max_norm / norm)
                        modified = True

            trace = bend.trace
            if tracer is None or trace is None or step % trace.sample_every != 0:
                continue
            metadata = {
                "bend": bend.name,
                "layer": layer_name,
                "localizability": bend_localizability_label(bend),
            }
            if "activation_norm" in trace.metrics:
                tracer.log(step=step, metric_name="activation_norm", value=float(current.norm().item()), metadata=metadata)
            if "activation_delta_norm" in trace.metrics:
                delta = current - (prev.to(device=current.device, dtype=current.dtype) if prev is not None else before)
                tracer.log(step=step, metric_name="activation_delta_norm", value=float(delta.norm().item()), metadata=metadata)

        return current if modified else None

    return _hook
