"""Flagship diffusion experiment for residual echo-chamber dynamics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import Field, model_validator

from neural_bending_toolkit.analysis.bend_tagging import score_and_tag_metastability
from neural_bending_toolkit.analysis.metastability_residual import (
    delta_norm_profile,
    echo_lock_in_index,
    plot_delta_norm_over_steps,
    recovery_index,
)
from neural_bending_toolkit.bends.v2 import BendPlan, BendSpec
from neural_bending_toolkit.bends.v2_diffusion_residual import compile_diffusion_residual_hook
from neural_bending_toolkit.experiment import Experiment, ExperimentSettings, RunContext


class ResidualEchoChamberDiffusionConfig(ExperimentSettings):
    model_id: str = "hf-internal-testing/tiny-stable-diffusion-pipe"
    prompt: str | list[str] = "a cinematic still-life of repeating geometric motifs"
    negative_prompt: str | None = None
    seed: int = 17
    num_inference_steps: int = Field(default=30, ge=1, le=200)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=30.0)
    samples_per_condition: int = Field(default=3, ge=1, le=32)
    echo: list[BendSpec] = Field(default_factory=list)
    counter: list[BendSpec] | None = None
    residual_layer_pattern: str = r"(?:resnets|transformer_blocks)"
    max_cache_layers: int = Field(default=64, ge=1)

    @model_validator(mode="after")
    def validate_echo(self) -> "ResidualEchoChamberDiffusionConfig":
        if not self.echo:
            raise ValueError("echo must include at least one BendSpec")
        return self


@dataclass
class _SampleRecord:
    image_array: np.ndarray


class _ContextTracer:
    def __init__(self, context: RunContext, metadata: dict[str, Any], collector: list[dict[str, Any]]) -> None:
        self.context = context
        self.metadata = metadata
        self.collector = collector

    def log(self, *, step: int, metric_name: str, value: float | int, metadata: dict[str, Any] | None = None) -> None:
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        self.collector.append({"step": step, "metric_name": metric_name, "value": float(value), "metadata": merged})
        self.context.log_metric(step=step, metric_name=metric_name, value=value, metadata=merged)


class ResidualEchoChamberDiffusion(Experiment):
    name = "residual-echo-chamber-diffusion"
    config_model = ResidualEchoChamberDiffusionConfig

    def _load_adapter(self):
        from neural_bending_toolkit.models.diffusion_diffusers import DiffusersStableDiffusionAdapter

        return DiffusersStableDiffusionAdapter(self.config.model_id, device="cpu")

    def run(self, context: RunContext) -> None:
        cfg = self.config
        adapter = self._load_adapter()
        prompts = [cfg.prompt] if isinstance(cfg.prompt, str) else list(cfg.prompt)
        conditions = {
            "baseline": [],
            "echo": list(cfg.echo),
            "echo_breaker": list(cfg.echo) + list(cfg.counter or []),
        }
        per_condition: dict[str, list[_SampleRecord]] = {k: [] for k in conditions}
        trace_rows: list[dict[str, Any]] = []

        for prompt_index, prompt in enumerate(prompts):
            for sample_index in range(cfg.samples_per_condition):
                seed = cfg.seed + sample_index
                for condition_name, bends in conditions.items():
                    hook = None
                    if bends:
                        hook = compile_diffusion_residual_hook(
                            BendPlan(bends=bends),
                            tracer=_ContextTracer(context, {"condition": condition_name, "prompt_index": prompt_index, "sample_index": sample_index}, trace_rows),
                        )
                    output = adapter.generate(
                        prompt,
                        negative_prompt=cfg.negative_prompt,
                        num_inference_steps=cfg.num_inference_steps,
                        guidance_scale=cfg.guidance_scale,
                        generator_seed=seed,
                        residual_hook=hook,
                        residual_layer_pattern=cfg.residual_layer_pattern,
                        max_cache_layers=cfg.max_cache_layers,
                    )
                    adapter.save_artifacts(output, context.run_dir / "conditions" / condition_name, prefix=f"prompt{prompt_index}_sample{sample_index}")
                    per_condition[condition_name].append(_SampleRecord(image_array=np.asarray(output.images[0]).astype(np.float32)))

        delta_profiles = {c: delta_norm_profile(trace_rows, condition=c) for c in conditions}
        start = min((b.site.timestep_start for b in cfg.echo if b.site.timestep_start is not None), default=0)
        end = max((b.site.timestep_end for b in cfg.echo if b.site.timestep_end is not None), default=start)
        lock = echo_lock_in_index(delta_profiles["echo"], pre_window=(0, max(start - 1, 0)), post_window=(end + 1, cfg.num_inference_steps - 1))
        recov = recovery_index(delta_profiles["echo_breaker"], breaker_window=(start, end), post_breaker_window=(end + 1, cfg.num_inference_steps - 1))

        baseline = per_condition["baseline"]
        diffs = {}
        for condition in ("echo", "echo_breaker"):
            mses = [float(np.mean((a.image_array - b.image_array) ** 2)) for a, b in zip(baseline, per_condition[condition], strict=True)]
            diffs[condition] = {"count": len(mses), "mean_mse": float(np.mean(mses)) if mses else 0.0}

        tagging = score_and_tag_metastability(
            recovery_index_value=float(recov),
            concentration_collapse_index_value=float(max(0.0, 1.0 - lock)),
            basin_shift_proxy={"pairwise_mse": [float(diffs.get("echo_breaker", {}).get("mean_mse", 0.0))]},
            output_variance=float(np.var([rec.image_array.mean() for recs in per_condition.values() for rec in recs])),
        )

        comparisons = {
            "baseline_vs_echo": diffs["echo"],
            "baseline_vs_echo_breaker": diffs["echo_breaker"],
            "residual_metrics": {"echo_lock_in_index": float(lock), "recovery_index": float(recov), "delta_norm_profile": {c: {str(k): v for k, v in p.items()} for c, p in delta_profiles.items()}},
            "tagging": tagging.model_dump(),
        }
        comparisons_dir = context.run_dir / "comparisons"
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        (comparisons_dir / "residual_metrics_comparison.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
        try:
            plot_delta_norm_over_steps(delta_profiles, comparisons_dir / "delta_norm_over_steps.png")
        except Exception:
            pass

        summary = {
            "conditions": {k: {"count": len(v)} for k, v in per_condition.items()},
            "tags": tagging.tags,
            "residual_metrics": comparisons["residual_metrics"],
            "artifacts": {"delta_norm_over_steps": str(comparisons_dir / "delta_norm_over_steps.png")},
        }
        (context.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
