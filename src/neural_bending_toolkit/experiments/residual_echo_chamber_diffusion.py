"""Flagship diffusion experiment for residual echo-chamber dynamics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import Field

from neural_bending_toolkit.analysis.metastability_residual import (
    activation_norm_profile,
    delta_norm_profile,
    echo_lock_in_index,
    novelty_proxy,
    plot_delta_norm_over_steps,
    recovery_index,
)
from neural_bending_toolkit.bends.v2 import BendPlan, BendSpec
from neural_bending_toolkit.bends.v2_diffusion_residual import (
    compile_diffusion_residual_hook,
)
from neural_bending_toolkit.experiment import Experiment, ExperimentSettings, RunContext


class ResidualEchoChamberDiffusionConfig(ExperimentSettings):
    """Configuration for residual echo-chamber diffusion benchmark."""

    model_id: str = "hf-internal-testing/tiny-stable-diffusion-pipe"
    prompt: str | list[str] = "a cinematic still-life of repeating geometric motifs"
    negative_prompt: str | None = None
    seed: int = 17
    num_inference_steps: int = Field(default=30, ge=1, le=200)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=30.0)
    samples_per_condition: int = Field(default=3, ge=1, le=32)

    residual_site_layer_regex: str = r"down_blocks\.[0-2]\..*(?:resnets|transformer_blocks)"
    residual_layer_pattern: str = r"(?:resnets|transformer_blocks)"
    max_layers_cached: int | None = Field(default=64, ge=1)

    echo_timestep_start: int = Field(default=10, ge=0)
    echo_timestep_end: int = Field(default=25, ge=0)
    echo_alpha_max: float = Field(default=0.25, ge=0.0, le=1.0)

    breaker_timestep_start: int = Field(default=26, ge=0)
    breaker_timestep_end: int = Field(default=30, ge=0)
    breaker_mode: Literal["residual_clamp", "residual_leak"] = "residual_clamp"
    breaker_clamp_max_norm: float = Field(default=18.0, gt=0.0)
    breaker_leak: float = Field(default=0.2, ge=0.0, le=1.0)
    breaker_noise_scale: float = Field(default=0.0, ge=0.0)


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
        self.condition = condition
        self._context = context
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
            "condition": self.condition,
            "prompt_index": self._prompt_index,
            "sample_index": self._sample_index,
        }
        if metadata is not None:
            combined.update(metadata)

        self._collector.append(
            {
                "step": step,
                "metric_name": metric_name,
                "value": float(value),
                "metadata": dict(combined),
            }
        )
        self._context.log_metric(
            step=step,
            metric_name=metric_name,
            value=value,
            metadata=combined,
        )


class ResidualEchoChamberDiffusion(Experiment):
    """Compare baseline, echo lock-in, and echo-breaker recovery conditions."""

    name = "residual-echo-chamber-diffusion"
    config_model = ResidualEchoChamberDiffusionConfig

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

    def _echo_bend(self) -> BendSpec:
        cfg = self.config
        return BendSpec.model_validate(
            {
                "name": "residual-echo-ramp",
                "site": {
                    "kind": "diffusion.residual",
                    "layer_regex": cfg.residual_site_layer_regex,
                    "timestep_start": cfg.echo_timestep_start,
                    "timestep_end": cfg.echo_timestep_end,
                },
                "actuator": {"type": "residual_echo", "params": {}},
                "schedule": {
                    "mode": "ramp",
                    "strength": cfg.echo_alpha_max,
                    "strength_start": 0.0,
                    "strength_end": cfg.echo_alpha_max,
                },
                "trace": {
                    "metrics": ["activation_norm", "activation_delta_norm"],
                    "sample_every": 1,
                },
            }
        )

    def _breaker_bend(self) -> BendSpec:
        cfg = self.config
        if cfg.breaker_mode == "residual_leak":
            actuator = {
                "type": "residual_leak",
                "params": {
                    "leak": cfg.breaker_leak,
                    "noise_scale": cfg.breaker_noise_scale,
                },
            }
        else:
            actuator = {
                "type": "residual_clamp",
                "params": {"max_norm": cfg.breaker_clamp_max_norm},
            }

        return BendSpec.model_validate(
            {
                "name": "echo-breaker-late-window",
                "site": {
                    "kind": "diffusion.residual",
                    "layer_regex": cfg.residual_site_layer_regex,
                    "timestep_start": cfg.breaker_timestep_start,
                    "timestep_end": cfg.breaker_timestep_end,
                },
                "actuator": actuator,
                "schedule": {"mode": "window", "strength": 1.0},
                "trace": {
                    "metrics": ["activation_norm", "activation_delta_norm"],
                    "sample_every": 1,
                },
            }
        )

    def run(self, context: RunContext) -> None:
        adapter = self._load_adapter()
        cfg = self.config
        prompts = self._as_prompt_list(cfg.prompt)

        conditions: dict[str, list[BendSpec]] = {
            "baseline": [],
            "echo": [self._echo_bend()],
            "echo_breaker": [self._echo_bend(), self._breaker_bend()],
        }

        per_condition: dict[str, list[_SampleRecord]] = {name: [] for name in conditions}
        trace_rows: list[dict[str, Any]] = []

        for prompt_index, prompt in enumerate(prompts):
            for sample_index in range(cfg.samples_per_condition):
                seed = cfg.seed + sample_index
                for condition_name, bends in conditions.items():
                    condition_dir = context.run_dir / "conditions" / condition_name
                    condition_dir.mkdir(parents=True, exist_ok=True)

                    residual_hook = None
                    if bends:
                        residual_hook = compile_diffusion_residual_hook(
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
                        residual_hook=residual_hook,
                        residual_layer_pattern=cfg.residual_layer_pattern,
                        max_layers_cached=cfg.max_layers_cached,
                    )
                    adapter.save_artifacts(
                        output,
                        condition_dir,
                        prefix=f"prompt{prompt_index}_sample{sample_index}",
                    )
                    per_condition[condition_name].append(
                        _SampleRecord(image_array=self._to_array(output.images[0]))
                    )

        def _paired_mse(left: str, right: str) -> dict[str, float | int]:
            left_records = per_condition[left]
            right_records = per_condition[right]
            mse_values = [
                self._mse(lhs.image_array, rhs.image_array)
                for lhs, rhs in zip(left_records, right_records, strict=True)
            ]
            return {
                "count": len(mse_values),
                "mean_mse": float(np.mean(mse_values)) if mse_values else 0.0,
                "max_mse": float(np.max(mse_values)) if mse_values else 0.0,
            }

        delta_profiles = {
            condition: delta_norm_profile(trace_rows, condition=condition)
            for condition in conditions
        }
        activation_profiles = {
            condition: activation_norm_profile(trace_rows, condition=condition)
            for condition in conditions
        }

        pre_window = (0, max(cfg.echo_timestep_start - 1, 0))
        post_window = (min(cfg.echo_timestep_end + 1, cfg.num_inference_steps - 1), cfg.num_inference_steps - 1)
        breaker_window = (cfg.breaker_timestep_start, cfg.breaker_timestep_end)
        post_breaker_window = (
            min(cfg.breaker_timestep_end + 1, cfg.num_inference_steps - 1),
            cfg.num_inference_steps - 1,
        )

        echo_lock = echo_lock_in_index(
            delta_profiles["echo"],
            pre_window=pre_window,
            post_window=post_window,
        )
        breaker_recovery = recovery_index(
            delta_profiles["echo_breaker"],
            breaker_window=breaker_window,
            post_breaker_window=post_breaker_window,
        )

        baseline_images = [r.image_array for r in per_condition["baseline"]]
        novelty = {
            condition: novelty_proxy(
                [r.image_array for r in records],
                baseline_images=baseline_images if condition != "baseline" else None,
            )
            for condition, records in per_condition.items()
        }

        comparisons = {
            "baseline_vs_echo": _paired_mse("baseline", "echo"),
            "baseline_vs_echo_breaker": _paired_mse("baseline", "echo_breaker"),
            "echo_vs_echo_breaker": _paired_mse("echo", "echo_breaker"),
            "residual_metrics": {
                "echo_lock_in_index": float(echo_lock),
                "recovery_index": float(breaker_recovery),
                "delta_norm_profile": {
                    condition: {str(step): value for step, value in profile.items()}
                    for condition, profile in delta_profiles.items()
                },
                "activation_norm_profile": {
                    condition: {str(step): value for step, value in profile.items()}
                    for condition, profile in activation_profiles.items()
                },
            },
            "novelty_proxy": novelty,
        }

        comparisons_dir = context.run_dir / "comparisons"
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = comparisons_dir / "residual_metrics_comparison.json"
        metrics_path.write_text(json.dumps(comparisons, indent=2), encoding="utf-8")

        delta_plot_path = comparisons_dir / "delta_norm_over_steps.png"
        delta_plot_written = False
        try:
            plot_delta_norm_over_steps(delta_profiles, delta_plot_path)
            delta_plot_written = True
        except Exception:
            pass

        tags = ["residual_echo_chamber"]
        if echo_lock < 1.0:
            tags.append("temporal_lock_in")
        if breaker_recovery > 0.0:
            tags.append("counter_coherence_reopening")

        summary = {
            "conditions": {
                name: {
                    "count": len(records),
                    "seed_start": cfg.seed,
                    "seed_end": cfg.seed + cfg.samples_per_condition - 1,
                }
                for name, records in per_condition.items()
            },
            "interpretation": {
                "echo": "temporal lock-in (repetition, motif fixation, novelty loss)",
                "echo_breaker": "counter-coherence reopening of late-step dynamics",
            },
            "tags": tags,
            "residual_metrics": comparisons["residual_metrics"],
            "novelty_proxy": novelty,
            "artifacts": {
                "residual_metrics_comparison": str(metrics_path),
                "delta_norm_over_steps": str(delta_plot_path) if delta_plot_written else None,
                "conditions_dir": str(context.run_dir / "conditions"),
            },
        }

        (context.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        context.post_intervention_snapshot(
            name="residual_echo_chamber_summary",
            data=summary,
        )
