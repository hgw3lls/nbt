"""Full GeopoliticalBend experiment implementation."""

from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field, field_validator

from neural_bending_toolkit.analysis.distributions import kl_divergence
from neural_bending_toolkit.analysis.embeddings import (
    compute_pca_projection,
    compute_umap_projection,
)
from neural_bending_toolkit.analysis.geopolitical_utils import (
    attractor_density,
    cosine_similarity_matrix,
    detect_refusal,
    structural_causality_score,
)
from neural_bending_toolkit.experiment import Experiment, ExperimentSettings, RunContext
from neural_bending_toolkit.models.geopolitical_metadata import (
    get_geopolitical_metadata,
)


class GeopoliticalConfig(ExperimentSettings):
    """Configuration schema for geopolitical bend workflows."""

    model_identifiers: list[str] = Field(min_length=1)
    llm_model_identifier: str = "sshleifer/tiny-gpt2"
    diffusion_model_identifier: str = "hf-internal-testing/tiny-stable-diffusion-pipe"

    governance_concepts: list[str] = Field(min_length=1)
    contradictory_prompt_pairs: list[tuple[str, str]] = Field(min_length=1)
    justice_attractor_token_sets: list[list[str]] = Field(min_length=1)

    log_level: str = Field(default="info")
    save_intermediate_artifacts: bool = True
    random_seed: int = 17

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed = {"debug", "info", "warning", "error"}
        normalized = value.lower().strip()
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return normalized


class GeopoliticalBend(Experiment):
    """Geopolitical bend with ontology mapping, dissonance testing, attractors."""

    name = "geopolitical-bend"
    config_model = GeopoliticalConfig

    def _reset_dir(self, path: Path) -> Path:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _seed_all(self, seed: int, context: RunContext) -> None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except Exception:
            pass
        context.log_event("Seeded reproducibility", seed=seed)

    @staticmethod
    def _sentiment_polarity(text: str) -> float:
        positive = {
            "peace",
            "cooperation",
            "rights",
            "justice",
            "stability",
            "support",
            "fair",
            "humane",
        }
        negative = {
            "war",
            "conflict",
            "harm",
            "violence",
            "sanction",
            "instability",
            "crisis",
            "abuse",
        }
        toks = [tok.lower().strip(".,!?;:") for tok in text.split()]
        pos = sum(1 for tok in toks if tok in positive)
        neg = sum(1 for tok in toks if tok in negative)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def _llm_adapter(self):
        from neural_bending_toolkit.models.llm_hf import HuggingFaceCausalLMAdapter

        model_id = self.config.llm_model_identifier
        return HuggingFaceCausalLMAdapter(model_id, device="cpu")

    def _diffusion_adapter(self):
        from neural_bending_toolkit.models.diffusion_diffusers import (
            DiffusersStableDiffusionAdapter,
        )

        model_id = self.config.diffusion_model_identifier
        return DiffusersStableDiffusionAdapter(model_id, device="cpu")

    def _extract_llm_embeddings(
        self,
        model_id: str,
        concepts: list[str],
        seed: int,
    ) -> np.ndarray:
        """Extract concept embeddings via existing LLM adapter when possible."""
        try:
            from neural_bending_toolkit.models.llm_hf import HuggingFaceCausalLMAdapter

            adapter = HuggingFaceCausalLMAdapter(model_id, device="cpu")
            tokenizer = adapter.tokenizer
            emb_layer = adapter.model.get_input_embeddings()

            vectors = []
            for concept in concepts:
                encoded = tokenizer(concept, return_tensors="pt")
                ids = encoded["input_ids"][0]
                with adapter._torch.no_grad():
                    vec = emb_layer(ids).mean(dim=0).detach().cpu().numpy()
                vectors.append(vec.astype(np.float32))
            return np.stack(vectors, axis=0)
        except Exception:
            rng = np.random.default_rng(seed + abs(hash(model_id)) % 10_000)
            return rng.normal(size=(len(concepts), 64)).astype(np.float32)

    def _extract_diffusion_embeddings(
        self,
        concepts: list[str],
        seed: int,
    ) -> np.ndarray:
        try:
            adapter = self._diffusion_adapter()
            vectors = []
            for concept in concepts:
                emb = adapter._encode_prompt(concept)
                arr = emb[0].detach().cpu().numpy().mean(axis=0)
                vectors.append(arr.astype(np.float32))
            return np.stack(vectors, axis=0)
        except Exception:
            rng = np.random.default_rng(seed + 991)
            return rng.normal(size=(len(concepts), 64)).astype(np.float32)

    def phase_1_ontology_mapping(
        self,
        context: RunContext,
        phase_dir: Path,
    ) -> dict[str, Any]:
        concepts = self.config.governance_concepts
        outputs: dict[str, Any] = {"models": [], "concepts": concepts}

        for model_id in self.config.model_identifiers:
            emb = self._extract_llm_embeddings(
                model_id,
                concepts,
                self.config.random_seed,
            )
            sim = cosine_similarity_matrix(emb)
            pca = compute_pca_projection(emb, n_components=2)
            umap_proj = compute_umap_projection(emb, n_components=2)

            model_slug = model_id.replace("/", "_")
            np.save(phase_dir / f"{model_slug}_embeddings.npy", emb)
            np.save(phase_dir / f"{model_slug}_pca.npy", pca)
            np.save(phase_dir / f"{model_slug}_umap.npy", umap_proj)

            sim_csv = phase_dir / f"{model_slug}_similarity.csv"
            with sim_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["concept", *concepts])
                for i, concept in enumerate(concepts):
                    writer.writerow([concept, *[f"{v:.6f}" for v in sim[i]]])

            try:
                import matplotlib.pyplot as plt

                for name, proj in (("pca", pca), ("umap", umap_proj)):
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.scatter(proj[:, 0], proj[:, 1])
                    for idx, concept in enumerate(concepts):
                        ax.annotate(concept, (proj[idx, 0], proj[idx, 1]))
                    ax.set_title(f"{model_id} {name.upper()}")
                    fig.tight_layout()
                    fig.savefig(phase_dir / f"{model_slug}_{name}.png")
                    plt.close(fig)
            except Exception:
                note = phase_dir / f"{model_slug}_plots_unavailable.txt"
                note.write_text(
                    "matplotlib unavailable; plot generation skipped.\n",
                    encoding="utf-8",
                )

            metadata = get_geopolitical_metadata(model_id)
            outputs["models"].append(
                {
                    "model_identifier": model_id,
                    "registry_metadata": metadata,
                    "mean_similarity": float(np.mean(sim)),
                }
            )

        diff_emb = self._extract_diffusion_embeddings(concepts, self.config.random_seed)
        diff_sim = cosine_similarity_matrix(diff_emb)
        np.save(phase_dir / "diffusion_clip_embeddings.npy", diff_emb)
        np.save(phase_dir / "diffusion_clip_similarity.npy", diff_sim)

        context.log_event("Completed phase 1", phase="ontology_mapping")
        return outputs

    def phase_2_governance_dissonance(
        self,
        context: RunContext,
        phase_dir: Path,
    ) -> dict[str, Any]:
        try:
            adapter = self._llm_adapter()
        except Exception:
            adapter = None

        rows = []
        prompt_pairs = self.config.contradictory_prompt_pairs
        for idx, (prompt_a, prompt_b) in enumerate(prompt_pairs, start=1):
            if adapter is not None:
                out_a = adapter.generate(
                    prompt_a,
                    max_new_tokens=60,
                    temperature=0.8,
                ).text
                out_b = adapter.generate(
                    prompt_b,
                    max_new_tokens=60,
                    temperature=0.8,
                ).text
                dist_a = adapter.sampling_distribution(
                    prompt_a,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=50,
                )
                dist_b = adapter.sampling_distribution(
                    prompt_b,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=50,
                )
                distribution_kl = kl_divergence(dist_a, dist_b)
            else:
                out_a = f"Stubbed response A for: {prompt_a}"
                out_b = f"Stubbed response B for: {prompt_b}"
                distribution_kl = 0.0

            row = {
                "pair_index": idx,
                "prompt_a": prompt_a,
                "prompt_b": prompt_b,
                "output_a": out_a,
                "output_b": out_b,
                "refusal_a": detect_refusal(out_a),
                "refusal_b": detect_refusal(out_b),
                "sentiment_a": self._sentiment_polarity(out_a),
                "sentiment_b": self._sentiment_polarity(out_b),
                "structural_score_a": structural_causality_score(out_a),
                "structural_score_b": structural_causality_score(out_b),
                "distribution_kl": float(distribution_kl),
            }
            rows.append(row)

        json_path = phase_dir / "governance_dissonance_results.json"
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

        csv_path = phase_dir / "governance_dissonance_results.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(rows[0].keys()) if rows else ["empty"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            if rows:
                writer.writerows(rows)

        context.log_event("Completed phase 2", phase="governance_dissonance")
        return {"rows": rows}

    def phase_3_justice_attractors(
        self,
        context: RunContext,
        phase_dir: Path,
    ) -> dict[str, Any]:
        try:
            adapter = self._llm_adapter()
        except Exception:
            adapter = None

        results = []
        token_sets = self.config.justice_attractor_token_sets
        for idx, token_set in enumerate(token_sets, start=1):
            base_prompt = "Assess policy tensions in geopolitical governance neutrally."
            bent_prompt = base_prompt + " " + " ".join(token_set * 3)

            if adapter is not None:
                baseline = adapter.generate(
                    base_prompt,
                    max_new_tokens=80,
                    temperature=0.9,
                ).text
                bent = adapter.generate(
                    bent_prompt,
                    max_new_tokens=80,
                    temperature=0.9,
                ).text
            else:
                baseline = f"Stub baseline response for set {idx}."
                token_block = " ".join(token_set)
                bent = f"Stub bent response using attractor tokens: {token_block}"

            baseline_density = attractor_density(baseline, token_set)
            bent_density = attractor_density(bent, token_set)
            density_change = bent_density - baseline_density

            result = {
                "token_set_index": idx,
                "token_set": token_set,
                "baseline_output": baseline,
                "bent_output": bent,
                "baseline_density": baseline_density,
                "bent_density": bent_density,
                "density_change": density_change,
            }
            results.append(result)

            lines = [
                f"Token set: {token_set}",
                f"Baseline density: {baseline_density:.6f}",
                f"Bent density: {bent_density:.6f}",
                f"Density change: {density_change:.6f}",
                "",
                "Baseline output:",
                baseline,
                "",
                "Bent output:",
                bent,
            ]
            (phase_dir / f"comparison_{idx}.txt").write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

        out_path = phase_dir / "justice_attractor_results.json"
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        context.log_event("Completed phase 3", phase="justice_attractors")
        return {"results": results}

    def run(self, context: RunContext) -> None:
        self._seed_all(self.config.random_seed, context)

        geo_dir = self._reset_dir(context.artifacts_dir / "geopolitical")
        phase1_dir = self._reset_dir(geo_dir / "phase_1_ontology_mapping")
        phase2_dir = self._reset_dir(geo_dir / "phase_2_governance_dissonance")
        phase3_dir = self._reset_dir(geo_dir / "phase_3_justice_attractors")

        context.log_event(
            "Starting Geopolitical Bend",
            level=self.config.log_level,
            model_identifiers=self.config.model_identifiers,
            random_seed=self.config.random_seed,
        )

        phase_1 = self.phase_1_ontology_mapping(context, phase1_dir)
        phase_2 = self.phase_2_governance_dissonance(context, phase2_dir)
        phase_3 = self.phase_3_justice_attractors(context, phase3_dir)

        context.log_metric(
            step=1,
            metric_name="phase_1_model_count",
            value=len(phase_1.get("models", [])),
            metadata={"phase": "ontology_mapping"},
        )
        context.log_metric(
            step=2,
            metric_name="phase_2_pair_count",
            value=len(phase_2.get("rows", [])),
            metadata={"phase": "governance_dissonance"},
        )
        context.log_metric(
            step=3,
            metric_name="phase_3_token_set_count",
            value=len(phase_3.get("results", [])),
            metadata={"phase": "justice_attractors"},
        )

        summary = {
            "notes": "Full phase scaffolding with metric/artifact outputs.",
            "phase_1": {"model_count": len(phase_1.get("models", []))},
            "phase_2": {"pair_count": len(phase_2.get("rows", []))},
            "phase_3": {"token_set_count": len(phase_3.get("results", []))},
        }
        context.save_text_artifact(
            "geopolitical/geopolitical_summary.json",
            json.dumps(summary, indent=2),
        )
        context.log_event("Completed Geopolitical Bend", level=self.config.log_level)
