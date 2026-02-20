"""Governance shock recovery experiment for diffusion pipelines."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

from neural_bending_toolkit.analysis.bend_tagging import (
    score_and_tag_metastability,
    write_comparison_report,
)
from neural_bending_toolkit.analysis.figures_metastability import plot_entropy_over_steps
from neural_bending_toolkit.analysis.metastability import (
    basin_shift_proxy,
    compute_attention_entropy_profile,
    compute_attention_topk_mass_profile,
    concentration_collapse_index,
    recovery_index,
)
from neural_bending_toolkit.bends.v2 import BendPlan, BendSpec
from neural_bending_toolkit.bends.v2_diffusion import compile_diffusion_cross_attention_hook
from neural_bending_toolkit.experiment import Experiment, ExperimentSettings, RunContext


class ComparisonFlags(BaseModel):
    """Toggle which condition comparisons are computed."""

    baseline_vs_shock: bool = True
    baseline_vs_shock_counter: bool = True
    shock_vs_shock_counter: bool = True


class GovernanceShockRecoveryDiffusionConfig(ExperimentSettings):
    """Configuration schema for governance shock recovery diffusion experiment."""

    model_id: str = "hf-internal-testing/tiny-stable-diffusion-pipe"
    prompt: str | list[str] = "a city council hearing in session"
    negative_prompt: str | None = None
    seed: int = 7
    num_inference_steps: int = Field(default=20, ge=1, le=200)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=30.0)
    height: int | None = Field(default=None, ge=64)
    width: int | None = Field(default=None, ge=64)
    samples_per_condition: int = Field(default=4, ge=1, le=32)
    shock: list[BendSpec] = Field(default_factory=list)
    counter: list[BendSpec] | None = None
    comparisons: ComparisonFlags = Field(default_factory=ComparisonFlags)

    @model_validator(mode="after")
    def validate_shock(self) -> GovernanceShockRecoveryDiffusionConfig:
        if not self.shock:
            raise ValueError("shock must include at least one BendSpec")
        return self


@dataclass
class _SampleRecord:
    condition: str
    prompt_index: int
    sample_index: int
    seed: int
    image_path: Path
    image_array: np.ndarray


class _ContextTracer:
    """Adapter that routes Bend v2 trace metrics into the run context."""

    def __init__(
        self,
        context: RunContext,
        metadata: dict[str, Any],
        collector: list[dict[str, Any]],
    ) -> None:
        self._context = context
        self._metadata = metadata
        self._collector = collector

    def log(
        self,
        *,
        step: int,
        metric_name: str,
        value: float | int,
        metadata: dict[str, object] | None = None,
    ) -> None:
        combined = dict(self._metadata)
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


class GovernanceShockRecoveryDiffusion(Experiment):
    """Compare baseline, shock, and shock+counter diffusion generations."""

    name = "governance-shock-recovery-diffusion"
    config_model = GovernanceShockRecoveryDiffusionConfig

    def _load_adapter(self):
        from neural_bending_toolkit.models.diffusion_diffusers import (
            DiffusersStableDiffusionAdapter,
        )

        return DiffusersStableDiffusionAdapter(self.config.model_id, device="cpu")

    @staticmethod
    def _as_prompt_list(prompt: str | list[str]) -> list[str]:
        if isinstance(prompt, str):
            return [prompt]
        return list(prompt)

    @staticmethod
    def _to_array(image: Any) -> np.ndarray:
        return np.asarray(image).astype(np.float32)

    @staticmethod
    def _mse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean((a - b) ** 2))

    def _compile_hook(
        self,
        *,
        bends: list[BendSpec],
        context: RunContext,
        metadata: dict[str, Any],
        collector: list[dict[str, Any]],
    ):
        if not bends:
            return None
        return compile_diffusion_cross_attention_hook(
            BendPlan(bends=bends),
            tracer=_ContextTracer(context, metadata=metadata, collector=collector),
        )

    @staticmethod
    def _shock_window(shock: list[BendSpec]) -> tuple[int, int]:
        starts = [bend.site.timestep_start for bend in shock if bend.site.timestep_start is not None]
        ends = [bend.site.timestep_end for bend in shock if bend.site.timestep_end is not None]
        if not starts:
            starts = [0]
        if not ends:
            ends = [max(starts[0], 0)]
        return min(starts), max(ends)


    @staticmethod
    def _output_variance(per_condition: dict[str, list[_SampleRecord]]) -> float:
        means: list[float] = []
        for records in per_condition.values():
            for record in records:
                arr = np.asarray(record.image_array, dtype=np.float32)
                if np.isnan(arr).any() or np.isinf(arr).any() or np.allclose(arr, 0.0):
                    return 0.0
                means.append(float(np.mean(arr)))
        if len(means) < 2:
            return 0.0
        return float(np.var(means))

    def _maybe_save_grid(self, records: list[_SampleRecord], destination: Path) -> bool:
        if not records:
            return False
        try:
            from PIL import Image
        except ModuleNotFoundError:
            return False

        first = records[0].image_array
        height, width = first.shape[0], first.shape[1]
        cols = 3
        rows = int(math.ceil(len(records) / cols))
        canvas = Image.new("RGB", (width * cols, height * rows))

        for idx, record in enumerate(records):
            arr = record.image_array
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            tile = Image.fromarray(arr)
            x = (idx % cols) * width
            y = (idx // cols) * height
            canvas.paste(tile, (x, y))

        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination)
        return True

    def run(self, context: RunContext) -> None:
        adapter = self._load_adapter()
        prompts = self._as_prompt_list(self.config.prompt)
        conditions_dir = context.run_dir / "conditions"

        conditions: dict[str, list[BendSpec]] = {
            "baseline": [],
            "shock": list(self.config.shock),
            "shock_counter": list(self.config.shock) + list(self.config.counter or []),
        }

        context.pre_intervention_snapshot(
            name="governance_shock_recovery_setup",
            data={
                "prompt_count": len(prompts),
                "samples_per_condition": self.config.samples_per_condition,
                "conditions": list(conditions.keys()),
            },
        )

        per_condition: dict[str, list[_SampleRecord]] = {key: [] for key in conditions}
        trace_rows: list[dict[str, Any]] = []

        for prompt_index, prompt in enumerate(prompts):
            for sample_index in range(self.config.samples_per_condition):
                seed = self.config.seed + sample_index
                for condition_name, bends in conditions.items():
                    condition_dir = conditions_dir / condition_name
                    condition_dir.mkdir(parents=True, exist_ok=True)

                    hook = self._compile_hook(
                        bends=bends,
                        context=context,
                        metadata={
                            "condition": condition_name,
                            "prompt_index": prompt_index,
                            "sample_index": sample_index,
                        },
                        collector=trace_rows,
                    )
                    output = adapter.generate(
                        prompt,
                        negative_prompt=self.config.negative_prompt,
                        num_inference_steps=self.config.num_inference_steps,
                        guidance_scale=self.config.guidance_scale,
                        generator_seed=seed,
                        cross_attention_hook=hook,
                    )
                    saved = adapter.save_artifacts(
                        output,
                        condition_dir,
                        prefix=f"prompt{prompt_index}_sample{sample_index}",
                    )
                    image_path = next(path for path in saved if path.suffix == ".png")
                    image_arr = self._to_array(output.images[0])
                    per_condition[condition_name].append(
                        _SampleRecord(
                            condition=condition_name,
                            prompt_index=prompt_index,
                            sample_index=sample_index,
                            seed=seed,
                            image_path=image_path,
                            image_array=image_arr,
                        )
                    )
                    context.log_event(
                        "Generated condition sample",
                        condition=condition_name,
                        prompt_index=prompt_index,
                        sample_index=sample_index,
                        seed=seed,
                        saved_count=len(saved),
                    )

        def _comparison_records(
            left: str,
            right: str,
            enabled: bool,
        ) -> dict[str, Any] | None:
            if not enabled:
                return None
            left_records = per_condition[left]
            right_records = per_condition[right]
            if len(left_records) != len(right_records):
                raise RuntimeError("Condition outputs are not aligned for comparison")
            mse_values = [
                self._mse(lhs.image_array, rhs.image_array)
                for lhs, rhs in zip(left_records, right_records, strict=True)
            ]
            return {
                "pair": f"{left}_vs_{right}",
                "count": len(mse_values),
                "mean_mse": float(np.mean(mse_values)) if mse_values else 0.0,
                "max_mse": float(np.max(mse_values)) if mse_values else 0.0,
                "min_mse": float(np.min(mse_values)) if mse_values else 0.0,
            }

        comparisons_payload: dict[str, Any] = {
            "baseline_vs_shock": _comparison_records(
                "baseline",
                "shock",
                self.config.comparisons.baseline_vs_shock,
            ),
            "baseline_vs_shock_counter": _comparison_records(
                "baseline",
                "shock_counter",
                self.config.comparisons.baseline_vs_shock_counter,
            ),
            "shock_vs_shock_counter": _comparison_records(
                "shock",
                "shock_counter",
                self.config.comparisons.shock_vs_shock_counter,
            ),
        }

        shock_window = self._shock_window(self.config.shock)
        metastability: dict[str, Any] = {}
        entropy_profiles: dict[str, list[float]] = {}
        for condition_name in conditions:
            condition_traces = [
                row
                for row in trace_rows
                if row.get("metadata", {}).get("condition") == condition_name
            ]
            entropy_profile = compute_attention_entropy_profile(condition_traces)
            topk_profile = compute_attention_topk_mass_profile(condition_traces)
            entropy_profiles[condition_name] = entropy_profile
            metastability[condition_name] = {
                "entropy_profile": entropy_profile,
                "topk_mass_profile": topk_profile,
                "recovery_index": recovery_index(entropy_profile, shock_window),
                "concentration_collapse_index": concentration_collapse_index(
                    topk_profile,
                    shock_window,
                ),
            }

        basin_proxy = basin_shift_proxy(
            {
                condition: [record.image_array for record in records]
                for condition, records in per_condition.items()
            }
        )
        metastability["basin_shift_proxy"] = basin_proxy

        output_variance = self._output_variance(per_condition)
        shock_counter_metrics = metastability.get("shock_counter", {})
        tagging = score_and_tag_metastability(
            recovery_index_value=float(shock_counter_metrics.get("recovery_index", 0.0)),
            concentration_collapse_index_value=float(
                shock_counter_metrics.get("concentration_collapse_index", 0.0)
            ),
            basin_shift_proxy=basin_proxy,
            output_variance=output_variance,
        )

        comparisons_payload["metastability"] = metastability
        comparisons_payload["tagging"] = tagging.model_dump()

        comparisons_dir = context.run_dir / "comparisons"
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = comparisons_dir / "metrics_comparison.json"
        metrics_path.write_text(
            json.dumps(comparisons_payload, indent=2),
            encoding="utf-8",
        )
        comparison_report_path = write_comparison_report(
            context.run_dir,
            comparisons=comparisons_payload,
            summary={"tags": tagging.model_dump()},
        )

        entropy_plot_path = comparisons_dir / "entropy_over_steps.png"
        entropy_plot_written = False
        try:
            plot_entropy_over_steps(entropy_profiles, entropy_plot_path)
            entropy_plot_written = True
        except Exception:
            context.log_event(
                "Skipping entropy-over-steps figure generation",
                reason="matplotlib unavailable or plotting failed",
            )

        baseline_ordered = {
            (record.prompt_index, record.sample_index): record for record in per_condition["baseline"]
        }
        grid_records: list[_SampleRecord] = []
        for condition_name in ("baseline", "shock", "shock_counter"):
            for record in per_condition[condition_name]:
                if condition_name == "baseline":
                    grid_records.append(record)
                    continue
                base_record = baseline_ordered[(record.prompt_index, record.sample_index)]
                grid_records.extend([base_record, record])

        grid_written = self._maybe_save_grid(grid_records, comparisons_dir / "image_grid.png")
        if not grid_written:
            context.log_event(
                "Skipping image grid generation",
                reason="Pillow unavailable or no records",
            )

        summary = {
            "conditions": {
                name: {
                    "count": len(records),
                    "seed_start": self.config.seed,
                    "seed_end": self.config.seed + self.config.samples_per_condition - 1,
                }
                for name, records in per_condition.items()
            },
            "comparisons": comparisons_payload,
            "tags": tagging.model_dump(),
            "metastability": {
                "shock_window": {"start": shock_window[0], "end": shock_window[1]},
                "recovery_index": {
                    condition: payload["recovery_index"]
                    for condition, payload in metastability.items()
                    if isinstance(payload, dict) and "recovery_index" in payload
                },
                "concentration_collapse_index": {
                    condition: payload["concentration_collapse_index"]
                    for condition, payload in metastability.items()
                    if isinstance(payload, dict)
                    and "concentration_collapse_index" in payload
                },
                "basin_shift_proxy": basin_proxy,
            },
            "artifacts": {
                "metrics_comparison": str(metrics_path),
                "comparison_report": str(comparison_report_path),
                "image_grid": str(comparisons_dir / "image_grid.png") if grid_written else None,
                "entropy_over_steps": str(entropy_plot_path) if entropy_plot_written else None,
            },
        }

        (context.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        context.post_intervention_snapshot(
            name="governance_shock_recovery_summary",
            data=summary,
        )
