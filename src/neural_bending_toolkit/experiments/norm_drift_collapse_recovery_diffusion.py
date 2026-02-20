"""Flagship diffusion experiment for normalization drift collapse/recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import Field

from neural_bending_toolkit.analysis.bend_tagging_norm import (
    score_and_tag_norm_metastability,
)
from neural_bending_toolkit.analysis.metastability_norm import (
    collapse_index,
    plot_norm_variance_over_steps,
    recovery_index,
    stability_proxy,
    variance_profile,
)
from neural_bending_toolkit.bends.v2 import BendPlan, BendSpec
from neural_bending_toolkit.bends.v2_diffusion_norm import compile_diffusion_norm_hook
from neural_bending_toolkit.experiment import Experiment, ExperimentSettings, RunContext


class NormDriftCollapseRecoveryDiffusionConfig(ExperimentSettings):
    """Configuration for norm drift collapse/recovery diffusion benchmark."""

    model_id: str = "hf-internal-testing/tiny-stable-diffusion-pipe"
    prompt: str | list[str] = "a detailed city skyline at dusk"
    negative_prompt: str | None = None
    seed: int = 11
    num_inference_steps: int = Field(default=24, ge=1, le=200)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=30.0)
    samples_per_condition: int = Field(default=3, ge=1, le=32)
    norm_layer_regex: str = "(norm|ln|layernorm)"

    collapse_timestep_start: int = Field(default=8, ge=0)
    collapse_timestep_end: int = Field(default=14, ge=0)
    recovery_timestep_start: int = Field(default=16, ge=0)
    recovery_timestep_end: int = Field(default=22, ge=0)

    collapse_gain_strength: float = 1.5
    collapse_clamp_strength: float = 1.0
    recovery_gain_strength: float = -0.35


@dataclass
class _SampleRecord:
    image_array: np.ndarray


class _ContextTracer:
    def __init__(
        self,
        context: RunContext,
        *,
        condition: str,
        prompt_index: int,
        sample_index: int,
        collector: list[dict[str, Any]],
    ) -> None:
        self._context = context
        self._condition = condition
        self._prompt_index = prompt_index
        self._sample_index = sample_index
        self._collector = collector

    def log(
        self,
        *,
        step: int,
        metric_name: str,
        value: float | int,
        metadata: dict[str, object] | None = None,
    ) -> None:
        combined = {
            "condition": self._condition,
            "prompt_index": self._prompt_index,
            "sample_index": self._sample_index,
        }
        if metadata is not None:
            combined.update(metadata)

        row = {
            "step": step,
            "metric_name": metric_name,
            "value": float(value),
            "metadata": dict(combined),
        }
        self._collector.append(row)
        self._context.log_metric(
            step=step,
            metric_name=metric_name,
            value=value,
            metadata=combined,
        )


class NormDriftCollapseRecoveryDiffusion(Experiment):
    """Evaluate norm-collapse and recovery bends over shared diffusion seeds."""

    name = "norm-drift-collapse-recovery-diffusion"
    config_model = NormDriftCollapseRecoveryDiffusionConfig

    def _load_adapter(self):
        from neural_bending_toolkit.models.diffusion_diffusers import (
            DiffusersStableDiffusionAdapter,
        )

        return DiffusersStableDiffusionAdapter(self.config.model_id, device="cpu")

    @staticmethod
    def _as_prompt_list(prompt: str | list[str]) -> list[str]:
        return [prompt] if isinstance(prompt, str) else list(prompt)

    @staticmethod
    def _to_array(image: Any) -> np.ndarray:
        return np.asarray(image).astype(np.float32)

    @staticmethod
    def _mse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean((a - b) ** 2))

    def _collapse_bends(self) -> list[BendSpec]:
        cfg = self.config
        return [
            BendSpec.model_validate(
                {
                    "name": "norm-collapse-gain",
                    "site": {
                        "kind": "diffusion.norm",
                        "layer_regex": cfg.norm_layer_regex,
                        "timestep_start": cfg.collapse_timestep_start,
                        "timestep_end": cfg.collapse_timestep_end,
                    },
                    "actuator": {"type": "norm_gain_drift", "params": {}},
                    "schedule": {"mode": "window", "strength": cfg.collapse_gain_strength},
                    "trace": {"metrics": ["norm_output_var"], "sample_every": 1},
                }
            ),
            BendSpec.model_validate(
                {
                    "name": "norm-collapse-clamp",
                    "site": {
                        "kind": "diffusion.norm",
                        "layer_regex": cfg.norm_layer_regex,
                        "timestep_start": cfg.collapse_timestep_start,
                        "timestep_end": cfg.collapse_timestep_end,
                    },
                    "actuator": {
                        "type": "norm_stat_clamp",
                        "params": {"min_var": 0.01, "max_var": 0.04},
                    },
                    "schedule": {"mode": "window", "strength": cfg.collapse_clamp_strength},
                    "trace": {"metrics": ["norm_output_var"], "sample_every": 1},
                }
            ),
        ]

    def _recovery_bends(self) -> list[BendSpec]:
        cfg = self.config
        bends = self._collapse_bends()
        bends.append(
            BendSpec.model_validate(
                {
                    "name": "norm-recovery-late-gain-counter",
                    "site": {
                        "kind": "diffusion.norm",
                        "layer_regex": cfg.norm_layer_regex,
                        "timestep_start": cfg.recovery_timestep_start,
                        "timestep_end": cfg.recovery_timestep_end,
                    },
                    "actuator": {"type": "norm_gain_drift", "params": {}},
                    "schedule": {"mode": "window", "strength": cfg.recovery_gain_strength},
                    "trace": {"metrics": ["norm_output_var"], "sample_every": 1},
                }
            )
        )
        return bends

    def run(self, context: RunContext) -> None:
        adapter = self._load_adapter()
        cfg = self.config
        prompts = self._as_prompt_list(cfg.prompt)

        conditions: dict[str, list[BendSpec]] = {
            "baseline": [],
            "norm_collapse": self._collapse_bends(),
            "norm_recovery": self._recovery_bends(),
        }

        per_condition: dict[str, list[_SampleRecord]] = {name: [] for name in conditions}
        trace_rows: list[dict[str, Any]] = []

        for prompt_index, prompt in enumerate(prompts):
            for sample_index in range(cfg.samples_per_condition):
                seed = cfg.seed + sample_index
                for condition_name, bends in conditions.items():
                    condition_dir = context.run_dir / "conditions" / condition_name
                    condition_dir.mkdir(parents=True, exist_ok=True)

                    norm_hook = None
                    if bends:
                        norm_hook = compile_diffusion_norm_hook(
                            BendPlan(bends=list(bends)),
                            tracer=_ContextTracer(
                                context,
                                condition=condition_name,
                                prompt_index=prompt_index,
                                sample_index=sample_index,
                                collector=trace_rows,
                            ),
                        )

                    output = adapter.generate(
                        prompt,
                        negative_prompt=cfg.negative_prompt,
                        num_inference_steps=cfg.num_inference_steps,
                        guidance_scale=cfg.guidance_scale,
                        generator_seed=seed,
                        cross_attention_hook=None,
                        norm_hook=norm_hook,
                    )
                    adapter.save_artifacts(
                        output,
                        condition_dir,
                        prefix=f"prompt{prompt_index}_sample{sample_index}",
                    )
                    per_condition[condition_name].append(
                        _SampleRecord(image_array=self._to_array(output.images[0]))
                    )

        collapse_window = (cfg.collapse_timestep_start, cfg.collapse_timestep_end)
        recovery_window = (cfg.recovery_timestep_start, cfg.recovery_timestep_end)
        pre_window = (0, max(collapse_window[0] - 1, 0))

        profiles = {
            condition: variance_profile(trace_rows, condition=condition)
            for condition in conditions
        }

        collapse_scores = {
            condition: collapse_index(
                profile,
                pre_window=pre_window,
                during_window=collapse_window,
            )
            for condition, profile in profiles.items()
            if condition != "baseline"
        }
        recovery_scores = {
            condition: recovery_index(
                profile,
                during_window=collapse_window,
                post_window=recovery_window,
            )
            for condition, profile in profiles.items()
            if condition != "baseline"
        }
        stability_scores = {
            condition: stability_proxy(profile)
            for condition, profile in profiles.items()
            if condition != "baseline"
        }

        baseline = per_condition["baseline"]
        diffs: dict[str, dict[str, float]] = {}
        for condition in ("norm_collapse", "norm_recovery"):
            mse_values = [
                self._mse(lhs.image_array, rhs.image_array)
                for lhs, rhs in zip(baseline, per_condition[condition], strict=True)
            ]
            diffs[condition] = {
                "count": len(mse_values),
                "mean_mse": float(np.mean(mse_values)) if mse_values else 0.0,
                "max_mse": float(np.max(mse_values)) if mse_values else 0.0,
            }

        tagging = score_and_tag_norm_metastability(
            collapse_index_value=float(collapse_scores.get("norm_recovery", 0.0)),
            recovery_index_value=float(recovery_scores.get("norm_recovery", 0.0)),
            stability_proxy_value=float(stability_scores.get("norm_recovery", 0.0)),
            final_basin_shift=float(diffs.get("norm_recovery", {}).get("mean_mse", 0.0)),
        )

        comparisons_dir = context.run_dir / "comparisons"
        comparisons_dir.mkdir(parents=True, exist_ok=True)

        metrics_payload = {
            "norm_output_var_profiles": {
                condition: {str(step): value for step, value in profile.items()}
                for condition, profile in profiles.items()
            },
            "collapse_index": collapse_scores,
            "recovery_index": recovery_scores,
            "stability_proxy": stability_scores,
            "image_difference_proxy": diffs,
            "tagging": tagging.model_dump(),
            "windows": {
                "pre": {"start": pre_window[0], "end": pre_window[1]},
                "collapse": {"start": collapse_window[0], "end": collapse_window[1]},
                "recovery": {"start": recovery_window[0], "end": recovery_window[1]},
            },
        }

        metrics_path = comparisons_dir / "norm_metrics_comparison.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        figure_path = comparisons_dir / "norm_variance_over_steps.png"
        figure_written = False
        try:
            plot_norm_variance_over_steps(profiles, figure_path)
            figure_written = True
        except Exception:
            context.log_event(
                "Skipping norm variance figure generation",
                reason="matplotlib unavailable or plotting failed",
            )

        summary = {
            "conditions": {
                condition: {
                    "count": len(records),
                    "seed_start": cfg.seed,
                    "seed_end": cfg.seed + cfg.samples_per_condition - 1,
                }
                for condition, records in per_condition.items()
            },
            "collapse_index": collapse_scores,
            "recovery_index": recovery_scores,
            "stability_proxy": stability_scores,
            "tags": tagging.tags,
            "tagging": tagging.model_dump(),
            "artifacts": {
                "norm_metrics_comparison": str(metrics_path),
                "norm_variance_over_steps": str(figure_path) if figure_written else None,
            },
        }
        (context.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
