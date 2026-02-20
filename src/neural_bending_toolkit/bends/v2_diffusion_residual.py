"""Diffusion residual-stream compilation for Bend v2 plans."""

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


class _ResidualHookContext(Protocol):
    layer_name: str
    step: int
    cache: dict[str, dict[str, Any]]


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


def _log_trace(
    *,
    tracer: _MetricTracer | None,
    metric_name: str,
    value: float,
    step: int,
    bend_name: str,
    layer_name: str,
    sample_every: int,
    localizability: str,
) -> None:
    if tracer is None or sample_every <= 0 or step % sample_every != 0:
        return

    tracer.log(
        step=step,
        metric_name=metric_name,
        value=float(value),
        metadata={
            "bend": bend_name,
            "layer": layer_name,
            "localizability": localizability,
        },
    )


def compile_diffusion_residual_hook(
    plan: BendPlan,
    tracer: _MetricTracer | None = None,
) -> Any:
    """Compile a BendPlan into a residual-stream hook callback."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required to compile diffusion residual hooks") from exc

    residual_bends = [bend for bend in plan.bends if bend.site.kind == "diffusion.residual"]
    if not residual_bends:
        return lambda x, _ctx: x

    def _hook(x: Any, ctx: _ResidualHookContext) -> Any:
        layer_name = str(getattr(ctx, "layer_name", ""))
        step = int(getattr(ctx, "step", 0))

        current = x
        for bend in residual_bends:
            if not _layer_matches(bend, layer_name) or not _step_matches(bend, step):
                continue

            strength = _schedule_strength(bend, step)
            if strength == 0:
                continue

            before = current
            params = bend.actuator.params

            if bend.actuator.type == "residual_echo":
                echo_cache = getattr(ctx, "cache", {}).get("residual_echo", {})
                prev = echo_cache.get(layer_name)
                if prev is not None:
                    alpha = float(params.get("alpha", strength))
                    current = current + alpha * prev.to(
                        device=current.device,
                        dtype=current.dtype,
                    )

            elif bend.actuator.type == "residual_leak":
                leak = float(params.get("leak", strength))
                leak = min(max(leak, 0.0), 1.0)
                sigma = float(params.get("noise_scale", 1.0))
                noise = torch.randn_like(current) * sigma
                current = (1.0 - leak) * current + leak * noise

            elif bend.actuator.type == "residual_clamp":
                max_norm = float(params.get("max_norm", strength))
                if max_norm > 0:
                    norm = current.norm()
                    max_norm_tensor = current.new_tensor(max_norm)
                    if norm > max_norm_tensor:
                        current = current * (max_norm_tensor / norm)

            trace = bend.trace
            if trace is None:
                continue
            localizability = bend_localizability_label(bend)

            if "activation_norm" in trace.metrics:
                _log_trace(
                    tracer=tracer,
                    metric_name="activation_norm",
                    value=float(current.norm().item()),
                    step=step,
                    bend_name=bend.name,
                    layer_name=layer_name,
                    sample_every=trace.sample_every,
                    localizability=localizability,
                )

            if "activation_delta_norm" in trace.metrics:
                delta = current - before
                _log_trace(
                    tracer=tracer,
                    metric_name="activation_delta_norm",
                    value=float(delta.norm().item()),
                    step=step,
                    bend_name=bend.name,
                    layer_name=layer_name,
                    sample_every=trace.sample_every,
                    localizability=localizability,
                )

        return current

    return _hook
