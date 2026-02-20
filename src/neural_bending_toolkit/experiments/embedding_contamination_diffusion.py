"""Embedding contamination experiment for diffusion pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import Field

from neural_bending_toolkit.bends.v2 import BendPlan, BendSpec
from neural_bending_toolkit.bends.v2_diffusion import (
    compile_diffusion_cross_attention_hook,
)
from neural_bending_toolkit.experiment import Experiment, ExperimentSettings, RunContext


def blend_embeddings(
    base: np.ndarray,
    contaminant: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend two embedding tensors with weight alpha for contaminant."""
    return ((1.0 - alpha) * base) + (alpha * contaminant)


@dataclass
class DiffusionSampleSummary:
    """Simple summary of generated sample comparisons."""

    mse_vs_baseline: float


class EmbeddingContaminationDiffusionConfig(ExperimentSettings):
    """Configuration for embedding contamination in diffusion generation."""

    model_id: str = "hf-internal-testing/tiny-stable-diffusion-pipe"
    base_prompt: str = "a clean laboratory bench"
    contaminant_prompt: str = "a chaotic graffiti wall"
    contamination_alpha: float = Field(default=0.25, ge=0.0, le=1.0)
    num_inference_steps: int = Field(default=10, ge=1, le=100)
    guidance_scale: float = Field(default=7.0, ge=1.0, le=20.0)
    seed: int = 7
    bends: list[BendSpec] | None = None


class _RunContextTracer:
    """Adapter exposing RunContext metric logging via tracer protocol."""

    def __init__(self, context: RunContext) -> None:
        self._context = context

    def log(
        self,
        *,
        step: int,
        metric_name: str,
        value: float | int,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._context.log_metric(
            step=step,
            metric_name=metric_name,
            value=value,
            metadata=None if metadata is None else dict(metadata),
        )


class EmbeddingContaminationDiffusion(Experiment):
    """Blend text embeddings and compare output images and attention maps."""

    name = "embedding-contamination-diffusion"
    config_model = EmbeddingContaminationDiffusionConfig

    def _load_adapter(self):
        from neural_bending_toolkit.models.diffusion_diffusers import (
            DiffusersStableDiffusionAdapter,
        )

        return DiffusersStableDiffusionAdapter(self.config.model_id, device="cpu")

    @staticmethod
    def _image_to_array(image) -> np.ndarray:
        return np.asarray(image).astype(np.float32)

    def run(self, context: RunContext) -> None:
        adapter = self._load_adapter()

        base_emb = adapter._encode_prompt(self.config.base_prompt)
        contaminant_emb = adapter._encode_prompt(self.config.contaminant_prompt)

        base_np = base_emb.detach().cpu().numpy()
        contaminant_np = contaminant_emb.detach().cpu().numpy()
        mixed_np = blend_embeddings(
            base_np,
            contaminant_np,
            self.config.contamination_alpha,
        )

        torch = adapter._torch
        mixed_emb = torch.from_numpy(mixed_np).to(adapter.device)

        context.pre_intervention_snapshot(
            name="embedding_blend",
            data={
                "alpha": self.config.contamination_alpha,
                "base_prompt": self.config.base_prompt,
                "contaminant_prompt": self.config.contaminant_prompt,
            },
        )

        bend_hook = None
        if self.config.bends:
            bend_hook = compile_diffusion_cross_attention_hook(
                BendPlan(bends=self.config.bends),
                tracer=_RunContextTracer(context),
            )

        baseline_output = adapter.generate(
            self.config.base_prompt,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator_seed=self.config.seed,
        )
        contaminated_output = adapter.generate(
            self.config.base_prompt,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator_seed=self.config.seed,
            embedding_hook=lambda _emb, _ctx: mixed_emb,
            cross_attention_hook=bend_hook,
        )

        base_arr = self._image_to_array(baseline_output.images[0])
        contaminated_arr = self._image_to_array(contaminated_output.images[0])
        mse = float(np.mean((base_arr - contaminated_arr) ** 2))
        summary = DiffusionSampleSummary(mse_vs_baseline=mse)

        context.log_metric(
            step=1,
            metric_name="image_mse_vs_baseline",
            value=summary.mse_vs_baseline,
            metadata={"alpha": self.config.contamination_alpha},
        )

        saved = []
        saved.extend(
            adapter.save_artifacts(
                baseline_output,
                context.artifacts_dir,
                prefix="baseline",
            )
        )
        saved.extend(
            adapter.save_artifacts(
                contaminated_output,
                context.artifacts_dir,
                prefix="contaminated",
            )
        )
        context.log_event(
            "Saved diffusion artifacts",
            saved_count=len(saved),
            image_pairs=1,
        )

        context.post_intervention_snapshot(
            name="embedding_contamination_outcome",
            data={"mse_vs_baseline": summary.mse_vs_baseline},
        )
