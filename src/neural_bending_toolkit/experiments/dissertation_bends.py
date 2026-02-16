"""Experiment families for dissertation bend taxonomy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from neural_bending_toolkit.bends.primitives import (
    AttentionConflictInjector,
    AttentionHeadScaler,
    AttractorSeeder,
    EmbeddingBlend,
    JusticeReweighter,
    LowProbSampler,
    NormStatPerturber,
    ResidualNoiseInjector,
)
from neural_bending_toolkit.experiment import Experiment, ExperimentSettings, RunContext


@dataclass
class ArchitectureObservation:
    architecture: str
    baseline_score: float
    bent_score: float
    delta: float
    qualitative_baseline: str
    qualitative_bent: str


class BendFamilyConfig(ExperimentSettings):
    seed: int = 101
    memo_title: str = "Theory memo"


class BendFamilyExperiment(Experiment):
    family_name: str = "bend-family"
    architectures: list[str] = []

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.config.seed)

    def _state(self, dim: int = 32) -> dict[str, Any]:
        rng = self._rng()
        return {
            "embedding": rng.normal(size=(dim,)).astype(np.float32),
            "contaminant_embedding": rng.normal(size=(dim,)).astype(np.float32),
            "sampler": {"temperature": 1.0, "top_p": 1.0},
            "attention": rng.random(size=(4, 4)).astype(np.float32),
            "residual": rng.normal(size=(dim,)).astype(np.float32),
            "norm_stats": rng.normal(size=(dim,)).astype(np.float32),
            "justice_weights": np.ones((8,), dtype=np.float32),
            "attractor": np.zeros((8,), dtype=np.float32),
        }

    def _score(self, state: dict[str, Any]) -> float:
        emb = np.asarray(state["embedding"]).mean()
        attn = np.asarray(state["attention"]).std()
        res = np.asarray(state["residual"]).std()
        return float(abs(emb) + attn + res)

    def _qualitative(self, architecture: str, bent: bool) -> str:
        mode = "bent" if bent else "baseline"
        descriptor = "instability" if bent else "coherence"
        return f"{architecture}: {mode} sample shows {descriptor}"

    def _run_architecture(self, architecture: str) -> ArchitectureObservation:
        state = self._state()
        baseline = self._score(state)
        bent_state = self.apply_bend(state)
        bent = self._score(bent_state)
        return ArchitectureObservation(
            architecture=architecture,
            baseline_score=baseline,
            bent_score=bent,
            delta=bent - baseline,
            qualitative_baseline=self._qualitative(architecture, bent=False),
            qualitative_bent=self._qualitative(architecture, bent=True),
        )

    def apply_bend(self, state: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _write_theory_memo(
        self,
        run_dir: Path,
        observations: list[ArchitectureObservation],
    ) -> Path:
        mean_delta = float(np.mean([obs.delta for obs in observations]))
        threshold_note = (
            "Bend remained below stability threshold."
            if abs(mean_delta) < 0.15
            else "Bend crossed stability threshold with visible output divergence."
        )
        lines = [
            f"# {self.config.memo_title}: {self.family_name}",
            "",
            "## Observed limit / threshold behavior",
            f"- Mean delta: {mean_delta:.4f}",
            f"- Threshold interpretation: {threshold_note}",
            "",
            "## Architecture notes",
        ]
        for obs in observations:
            lines.append(
                f"- **{obs.architecture}** baseline={obs.baseline_score:.4f}, "
                f"bent={obs.bent_score:.4f}, delta={obs.delta:.4f}"
            )
        memo_path = run_dir / "theory_memo.md"
        memo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return memo_path

    def run(self, context: RunContext) -> None:
        observations = [self._run_architecture(arch) for arch in self.architectures]

        comparison = {
            "family": self.family_name,
            "baseline_vs_bent": [obs.__dict__ for obs in observations],
        }
        context.save_text_artifact(
            f"{self.family_name}_comparison.json",
            json.dumps(comparison, indent=2),
        )

        for idx, obs in enumerate(observations, start=1):
            context.log_metric(
                step=idx,
                metric_name=f"{self.family_name}_delta",
                value=obs.delta,
                metadata={"architecture": obs.architecture},
            )
            context.save_text_artifact(
                f"samples/{self.family_name}_{obs.architecture}.txt",
                f"baseline: {obs.qualitative_baseline}\n"
                f"bent: {obs.qualitative_bent}\n",
            )

        memo_path = self._write_theory_memo(context.run_dir, observations)
        context.log_event(
            "Generated theory memo",
            family=self.family_name,
            memo=str(memo_path),
        )


class EmbeddingContaminationFamilyConfig(BendFamilyConfig):
    alpha: float = Field(default=0.35, ge=0.0, le=1.0)


class EmbeddingContaminationFamily(BendFamilyExperiment):
    """Embedding Contamination: LLM embeddings + diffusion CLIP embeddings."""

    name = "family-embedding-contamination"
    family_name = "embedding_contamination"
    config_model = EmbeddingContaminationFamilyConfig
    architectures = ["llm_embeddings", "diffusion_clip_embeddings"]

    def apply_bend(self, state: dict[str, Any]) -> dict[str, Any]:
        primitive = EmbeddingBlend(alpha=self.config.alpha)
        return primitive.apply(state)


class CorpusStratigraphyFamilyConfig(BendFamilyConfig):
    temperature: float = Field(default=0.7, gt=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)


class CorpusStratigraphyFamily(BendFamilyExperiment):
    """Corpus Stratigraphy: LLM sampling + diffusion guidance scale analog."""

    name = "family-corpus-stratigraphy"
    family_name = "corpus_stratigraphy"
    config_model = CorpusStratigraphyFamilyConfig
    architectures = ["llm_sampling", "diffusion_guidance"]

    def apply_bend(self, state: dict[str, Any]) -> dict[str, Any]:
        primitive = LowProbSampler(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        out = primitive.apply(state)
        out["residual"] = np.asarray(out["residual"]) * (1.0 / self.config.temperature)
        return out


class InterHeadDriftFamilyConfig(BendFamilyConfig):
    scale: float = Field(default=1.6, ge=0.0, le=4.0)


class InterHeadDriftFamily(BendFamilyExperiment):
    """Inter-Head Drift: LLM attention + ViT attention perturbation."""

    name = "family-inter-head-drift"
    family_name = "inter_head_drift"
    config_model = InterHeadDriftFamilyConfig
    architectures = ["llm_attention", "vit_attention"]

    def apply_bend(self, state: dict[str, Any]) -> dict[str, Any]:
        return AttentionHeadScaler(scale=self.config.scale).apply(state)


class GovernanceDissonanceFamilyConfig(BendFamilyConfig):
    conflict_strength: float = Field(default=0.5, ge=0.0, le=1.0)


class GovernanceDissonanceFamily(BendFamilyExperiment):
    """Governance Dissonance: LLM and diffusion attention conflict injection."""

    name = "family-governance-dissonance"
    family_name = "governance_dissonance"
    config_model = GovernanceDissonanceFamilyConfig
    architectures = ["llm_attention_conflict", "diffusion_cross_attention_conflict"]

    def apply_bend(self, state: dict[str, Any]) -> dict[str, Any]:
        return AttentionConflictInjector(self.config.conflict_strength).apply(state)


class ResidualDistortionFamilyConfig(BendFamilyConfig):
    noise_std: float = Field(default=0.12, ge=0.0, le=0.5)


class ResidualDistortionFamily(BendFamilyExperiment):
    """Residual Distortion: LLM residual noise + diffusion denoise interruption."""

    name = "family-residual-distortion"
    family_name = "residual_distortion"
    config_model = ResidualDistortionFamilyConfig
    architectures = ["llm_residual", "diffusion_denoise_steps"]

    def apply_bend(self, state: dict[str, Any]) -> dict[str, Any]:
        primitive = ResidualNoiseInjector(
            self.config.noise_std,
            seed=self.config.seed,
        )
        return primitive.apply(state)


class NormPerturbationFamilyConfig(BendFamilyConfig):
    stat_shift: float = Field(default=0.4, ge=-2.0, le=2.0)


class NormPerturbationFamily(BendFamilyExperiment):
    """Norm Perturbation: transformer norm swap + GAN stat perturbation."""

    name = "family-norm-perturbation"
    family_name = "norm_perturbation"
    config_model = NormPerturbationFamilyConfig
    architectures = ["transformer_norm", "gan_norm_stats"]

    def apply_bend(self, state: dict[str, Any]) -> dict[str, Any]:
        return NormStatPerturber(self.config.stat_shift).apply(state)


class JusticeReweightingFamilyConfig(BendFamilyConfig):
    multiplier: float = Field(default=1.5, ge=0.1, le=4.0)


class JusticeReweightingFamily(BendFamilyExperiment):
    """Justice Reweighting: LLM attention reweight + diffusion guidance reweight."""

    name = "family-justice-reweighting"
    family_name = "justice_reweighting"
    config_model = JusticeReweightingFamilyConfig
    architectures = ["llm_attention_reweight", "diffusion_guidance_reweight"]

    def apply_bend(self, state: dict[str, Any]) -> dict[str, Any]:
        base = np.asarray(state["justice_weights"], dtype=np.float32)
        return JusticeReweighter(base * self.config.multiplier).apply(state)


class JusticeAttractorsFamilyConfig(BendFamilyConfig):
    attractor_scale: float = Field(default=0.8, ge=0.0, le=2.0)


class JusticeAttractorsFamily(BendFamilyExperiment):
    """Justice Attractors: LLM adapter seed + diffusion conditioning persistence."""

    name = "family-justice-attractors"
    family_name = "justice_attractors"
    config_model = JusticeAttractorsFamilyConfig
    architectures = ["llm_adapter_seed", "diffusion_conditioning_persistence"]

    def apply_bend(self, state: dict[str, Any]) -> dict[str, Any]:
        seed = np.ones((8,), dtype=np.float32) * self.config.attractor_scale
        return AttractorSeeder(seed).apply(state)
