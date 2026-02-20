"""Flagship diffusion experiment for normalization drift collapse/recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import Field, model_validator

from neural_bending_toolkit.analysis.bend_tagging import score_and_tag_metastability
from neural_bending_toolkit.analysis.metastability_norm import (
    collapse_index,
    plot_norm_variance_over_steps,
    recovery_index,
    variance_profile,
)
from neural_bending_toolkit.bends.v2 import BendPlan, BendSpec
from neural_bending_toolkit.bends.v2_diffusion_norm import compile_diffusion_norm_hook
from neural_bending_toolkit.experiment import Experiment, ExperimentSettings, RunContext


class NormDriftCollapseRecoveryDiffusionConfig(ExperimentSettings):
    model_id: str = "hf-internal-testing/tiny-stable-diffusion-pipe"
    prompt: str | list[str] = "a detailed city skyline at dusk"
    negative_prompt: str | None = None
    seed: int = 11
    num_inference_steps: int = Field(default=30, ge=1, le=200)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=30.0)
    samples_per_condition: int = Field(default=3, ge=1, le=32)
    collapse: list[BendSpec] = Field(default_factory=list)
    counter: list[BendSpec] | None = None

    @model_validator(mode="after")
    def validate_bends(self) -> "NormDriftCollapseRecoveryDiffusionConfig":
        if not self.collapse:
            raise ValueError("collapse must include at least one BendSpec")
        return self


@dataclass
class _SampleRecord:
    image_array: np.ndarray


class _ContextTracer:
    def __init__(self, context: RunContext, metadata: dict[str, Any], collector: list[dict[str, Any]]) -> None:
        self._context = context
        self._metadata = metadata
        self._collector = collector

    def log(self, *, step: int, metric_name: str, value: float | int, metadata: dict[str, Any] | None = None) -> None:
        merged = dict(self._metadata)
        if metadata:
            merged.update(metadata)
        row = {"step": step, "metric_name": metric_name, "value": float(value), "metadata": merged}
        self._collector.append(row)
        self._context.log_metric(step=step, metric_name=metric_name, value=value, metadata=merged)


class NormDriftCollapseRecoveryDiffusion(Experiment):
    name = "norm-drift-collapse-recovery-diffusion"
    config_model = NormDriftCollapseRecoveryDiffusionConfig

    def _load_adapter(self):
        from neural_bending_toolkit.models.diffusion_diffusers import DiffusersStableDiffusionAdapter

        return DiffusersStableDiffusionAdapter(self.config.model_id, device="cpu")

    def run(self, context: RunContext) -> None:
        cfg = self.config
        adapter = self._load_adapter()
        prompts = [cfg.prompt] if isinstance(cfg.prompt, str) else list(cfg.prompt)

        conditions: dict[str, list[BendSpec]] = {
            "baseline": [],
            "norm_collapse": list(cfg.collapse),
            "norm_recovery": list(cfg.collapse) + list(cfg.counter or []),
        }

        per_condition: dict[str, list[_SampleRecord]] = {k: [] for k in conditions}
        trace_rows: list[dict[str, Any]] = []

        for prompt_index, prompt in enumerate(prompts):
            for sample_index in range(cfg.samples_per_condition):
                seed = cfg.seed + sample_index
                for condition_name, bends in conditions.items():
                    hook = None
                    if bends:
                        hook = compile_diffusion_norm_hook(
                            BendPlan(bends=bends),
                            tracer=_ContextTracer(
                                context,
                                {"condition": condition_name, "prompt_index": prompt_index, "sample_index": sample_index},
                                trace_rows,
                            ),
                        )
                    output = adapter.generate(
                        prompt,
                        negative_prompt=cfg.negative_prompt,
                        num_inference_steps=cfg.num_inference_steps,
                        guidance_scale=cfg.guidance_scale,
                        generator_seed=seed,
                        norm_hook=hook,
                    )
                    adapter.save_artifacts(output, context.run_dir / "conditions" / condition_name, prefix=f"prompt{prompt_index}_sample{sample_index}")
                    per_condition[condition_name].append(_SampleRecord(image_array=np.asarray(output.images[0]).astype(np.float32)))

        profiles = {name: variance_profile(trace_rows, condition=name) for name in conditions}
        steps = [b.site.timestep_start for b in cfg.collapse if b.site.timestep_start is not None]
        ends = [b.site.timestep_end for b in cfg.collapse if b.site.timestep_end is not None]
        collapse_window = (min(steps) if steps else 0, max(ends) if ends else 0)
        recovery_window = (collapse_window[1] + 1, cfg.num_inference_steps - 1)
        pre_window = (0, max(collapse_window[0] - 1, 0))

        collapse_scores = {k: collapse_index(v, pre_window=pre_window, during_window=collapse_window) for k, v in profiles.items() if k != "baseline"}
        recovery_scores = {k: recovery_index(v, during_window=collapse_window, post_window=recovery_window) for k, v in profiles.items() if k != "baseline"}

        baseline = per_condition["baseline"]
        diffs = {}
        for condition in ("norm_collapse", "norm_recovery"):
            mses = [float(np.mean((a.image_array - b.image_array) ** 2)) for a, b in zip(baseline, per_condition[condition], strict=True)]
            diffs[condition] = {"count": len(mses), "mean_mse": float(np.mean(mses)) if mses else 0.0}

        tagging = score_and_tag_metastability(
            recovery_index_value=float(recovery_scores.get("norm_recovery", 0.0)),
            concentration_collapse_index_value=float(collapse_scores.get("norm_recovery", 0.0)),
            basin_shift_proxy={"pairwise_mse": [float(diffs.get("norm_recovery", {}).get("mean_mse", 0.0))]},
            output_variance=float(np.var([rec.image_array.mean() for recs in per_condition.values() for rec in recs])),
        )

        comparisons_dir = context.run_dir / "comparisons"
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        comparisons = {
            "norm_output_var_profiles": {c: {str(k): v for k, v in p.items()} for c, p in profiles.items()},
            "collapse_index": collapse_scores,
            "recovery_index": recovery_scores,
            "image_difference_proxy": diffs,
            "tagging": tagging.model_dump(),
        }
        (comparisons_dir / "norm_metrics_comparison.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")

        try:
            plot_norm_variance_over_steps(profiles, comparisons_dir / "norm_variance_over_steps.png")
        except Exception:
            pass

        summary = {
            "conditions": {k: {"count": len(v)} for k, v in per_condition.items()},
            "collapse_index": collapse_scores,
            "recovery_index": recovery_scores,
            "tags": tagging.tags,
            "tagging": tagging.model_dump(),
        }
        (context.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
