"""Automatic theory memo generation aligned to the bend rubric."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from neural_bending_toolkit.analysis.bend_classifier import write_bend_classification
from neural_bending_toolkit.analysis.derived_metrics import write_derived_metrics

try:
    from jinja2 import Template
except Exception:  # pragma: no cover - fallback path for minimal environments
    Template = None


class MemoEvidenceMetric(BaseModel):
    metric_name: str
    value: float
    normalized: float


class MemoEvidencePair(BaseModel):
    baseline_path: str
    bent_path: str
    score: float
    reason: str


class TheoryMemoContext(BaseModel):
    run_id: str
    bend_name: str
    experiment_name: str
    timestamp: str
    research_question: str
    limit_1: str
    limit_2: str
    threshold_signal: str
    threshold_point: str
    counter_coherence_notes: str
    artifact_ref_1: str
    artifact_ref_2: str
    next_iteration_plan: str

    metric_evidence: list[MemoEvidenceMetric] = Field(default_factory=list)
    qualitative_evidence: list[MemoEvidencePair] = Field(default_factory=list)
    figure_paths: list[str] = Field(default_factory=list)

    confidence_limit: str
    confidence_threshold: str
    confidence_bend: str
    confidence_counter_coherence: str
    edit_notes: list[str] = Field(default_factory=list)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _load_or_generate_derived(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "analysis" / "derived_metrics.json"
    if not path.exists():
        path = write_derived_metrics(run_dir)
    return _read_json(path)


def _load_or_generate_classification(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "analysis" / "bend_classification.json"
    if not path.exists():
        path = write_bend_classification(run_dir)
    return _read_json(path)


def _find_figure_paths(run_dir: Path) -> list[str]:
    candidates = []
    for pattern in ["analysis/**/*.png", "artifacts/**/*.png", "analysis/**/*.svg"]:
        candidates.extend(sorted(run_dir.glob(pattern)))
    return [str(path.relative_to(run_dir)) for path in candidates[:8]]


def _deterministic_top_metrics(
    derived: dict[str, Any], top_k: int = 3
) -> list[MemoEvidenceMetric]:
    normalized = derived.get("normalized", {})
    keys = [
        "divergence",
        "entropy_delta",
        "coherence_delta",
        "refusal_rate_delta",
        "attractor_density_delta",
        "structural_causality_delta",
        "cross_task_consistency",
    ]

    scored: list[tuple[str, float, float]] = []
    for key in keys:
        raw = derived.get(key)
        norm = normalized.get(key)
        if isinstance(raw, (int, float)) and isinstance(norm, (int, float)):
            scored.append((key, float(raw), float(norm)))

    scored.sort(key=lambda item: (-abs(item[2]), item[0]))
    return [
        MemoEvidenceMetric(metric_name=name, value=value, normalized=norm)
        for name, value, norm in scored[:top_k]
    ]


def _score_pair(path: Path) -> tuple[float, str]:
    suffix_score = 0.0
    lowered = path.name.lower()
    if "diverg" in lowered:
        suffix_score += 1.5
    if "attractor" in lowered or "density" in lowered:
        suffix_score += 1.2
    if "delta" in lowered:
        suffix_score += 0.8
    return suffix_score, "filename-signal"


def _collect_qualitative_pairs(
    run_dir: Path, max_pairs: int = 2
) -> list[MemoEvidencePair]:
    artifacts_dir = run_dir / "artifacts"
    if not artifacts_dir.exists():
        return []

    bent_candidates = sorted(
        [
            path
            for path in artifacts_dir.glob("**/*")
            if path.is_file() and "bent" in path.name.lower()
        ]
    )
    baseline_candidates = sorted(
        [
            path
            for path in artifacts_dir.glob("**/*")
            if path.is_file() and "baseline" in path.name.lower()
        ]
    )

    if not bent_candidates or not baseline_candidates:
        text_files = sorted(
            path for path in artifacts_dir.glob("**/*.txt") if path.is_file()
        )
        if len(text_files) >= 2:
            return [
                MemoEvidencePair(
                    baseline_path=str(text_files[0].relative_to(run_dir)),
                    bent_path=str(text_files[1].relative_to(run_dir)),
                    score=0.1,
                    reason="text-order-fallback",
                )
            ]
        return []

    scored_pairs: list[MemoEvidencePair] = []
    for bent in bent_candidates:
        key = bent.name.lower().replace("bent", "").replace("_", "")
        matched = None
        for base in baseline_candidates:
            base_key = base.name.lower().replace("baseline", "").replace("_", "")
            if key == base_key or key in base_key or base_key in key:
                matched = base
                break
        if matched is None:
            matched = baseline_candidates[0]

        score, reason = _score_pair(bent)
        scored_pairs.append(
            MemoEvidencePair(
                baseline_path=str(matched.relative_to(run_dir)),
                bent_path=str(bent.relative_to(run_dir)),
                score=score,
                reason=reason,
            )
        )

    scored_pairs.sort(
        key=lambda item: (-item.score, item.bent_path, item.baseline_path)
    )
    return scored_pairs[:max_pairs]


def _confidence_label(
    *,
    availability: dict[str, Any],
    normalized: dict[str, Any],
    keys: list[str],
) -> str:
    present = sum(1 for key in keys if availability.get(key) is True)
    strengths = [
        abs(float(normalized.get(key, 0.0)))
        for key in keys
        if isinstance(normalized.get(key), (int, float))
    ]
    max_strength = max(strengths) if strengths else 0.0

    if present == len(keys) and max_strength >= 0.8:
        return "high"
    if present >= 1 and max_strength >= 0.3:
        return "medium"
    return "low"


def _render_template(template_text: str, context: TheoryMemoContext) -> str:
    payload = context.model_dump()
    if Template is not None:
        return Template(template_text).render(**payload)

    rendered = template_text
    for key, value in payload.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered


def _markdown_enrichment(context: TheoryMemoContext) -> str:
    metric_lines = [
        (
            f"- `{item.metric_name}`: raw={item.value:.4f}, "
            f"normalized={item.normalized:.4f}"
        )
        for item in context.metric_evidence
    ] or ["- No strong quantitative evidence was available."]

    qualitative_lines = [
        (
            f"- Baseline `{item.baseline_path}` vs Bent `{item.bent_path}` "
            f"(score={item.score:.2f}, reason={item.reason})"
        )
        for item in context.qualitative_evidence
    ] or ["- No baseline/bent qualitative pairs were found."]

    figure_lines = [f"- `{path}`" for path in context.figure_paths] or [
        "- No figures detected."
    ]

    note_lines = [f"- {line}" for line in context.edit_notes] or ["- No edit notes."]

    return "\n".join(
        [
            "",
            "## Limit → Threshold → Bend → Counter-Coherence",
            "",
            (
                "This memo follows the rubric by identifying an observed limit, "
                "estimating threshold behavior, characterizing bend type, and "
                "recording counter-coherence constraints in one narrative thread."
            ),
            "",
            "## Quantitative evidence selection (deterministic)",
            *metric_lines,
            "",
            "## Qualitative evidence selection (deterministic)",
            *qualitative_lines,
            "",
            "## Confidence by section",
            f"- Observed limits: **{context.confidence_limit}**",
            f"- Threshold behavior: **{context.confidence_threshold}**",
            f"- Bend interpretation: **{context.confidence_bend}**",
            f"- Counter-coherence: **{context.confidence_counter_coherence}**",
            "",
            "## Figure paths",
            *figure_lines,
            "",
            "## Edit notes",
            *note_lines,
            "",
        ]
    )


def build_theory_memo(run_dir: Path, *, seed: int = 7) -> Path:
    """Generate `theory_memo.md` for a run directory."""
    del seed  # deterministic policy is sorting-based; seed kept for stable API.
    run_dir = Path(run_dir)

    config = _read_yaml(run_dir / "config.yaml")
    derived = _load_or_generate_derived(run_dir)
    classification = _load_or_generate_classification(run_dir)

    availability = derived.get("availability", {})
    normalized = derived.get("normalized", {})
    metric_evidence = _deterministic_top_metrics(derived, top_k=3)
    qualitative_pairs = _collect_qualitative_pairs(run_dir, max_pairs=2)
    figure_paths = _find_figure_paths(run_dir)

    bend_tag = str(classification.get("bend_tag", "unknown"))
    experiment_name = str(
        config.get("experiment", config.get("name", "unknown-experiment"))
    )

    limit_conf = _confidence_label(
        availability=availability,
        normalized=normalized,
        keys=["divergence", "entropy_delta"],
    )
    threshold_conf = _confidence_label(
        availability=availability,
        normalized=normalized,
        keys=["coherence_delta", "refusal_rate_delta"],
    )
    bend_conf = _confidence_label(
        availability=availability,
        normalized=normalized,
        keys=["structural_causality_delta", "attractor_density_delta"],
    )
    counter_conf = _confidence_label(
        availability=availability,
        normalized=normalized,
        keys=["cross_task_consistency", "coherence_delta"],
    )

    primary_metric = metric_evidence[0].metric_name if metric_evidence else "divergence"
    threshold_metric = (
        metric_evidence[1].metric_name if len(metric_evidence) > 1 else primary_metric
    )

    artifact_ref_1 = (
        qualitative_pairs[0].baseline_path if qualitative_pairs else "artifacts/"
    )
    artifact_ref_2 = (
        qualitative_pairs[0].bent_path if qualitative_pairs else "analysis/"
    )

    edit_notes = [
        "Inspect `analysis/derived_metrics.json` for missing metric fields.",
        "Cross-check `analysis/bend_classification.json` scoring and gate decisions.",
    ]
    if figure_paths:
        edit_notes.append(f"Review figure assets: {', '.join(figure_paths[:3])}.")
    if qualitative_pairs:
        top_pair = qualitative_pairs[0]
        edit_notes.append(
            "Read the highest-priority qualitative pair: "
            f"{top_pair.baseline_path} vs {top_pair.bent_path}."
        )

    context = TheoryMemoContext(
        run_id=run_dir.name,
        bend_name=bend_tag,
        experiment_name=experiment_name,
        timestamp=str(derived.get("generated_at", "unknown")),
        research_question=(
            "Which representational limits are crossed at threshold and how does "
            "the resulting bend reorganize coherence under intervention?"
        ),
        limit_1=(
            f"Primary limit evidence is `{primary_metric}`, indicating where baseline "
            "and bent behavior begin to separate under intervention."
        ),
        limit_2=(
            "Secondary evidence indicates pressure on response stability/refusal "
            "behavior under the configured prompt and model regime."
        ),
        threshold_signal=(
            f"The key transition signal is `{threshold_metric}` combined with "
            "coherence and refusal deltas."
        ),
        threshold_point=(
            "Transition appears when normalized effect size exceeds approximately "
            "0.3 and gate patterns begin favoring one bend family."
        ),
        counter_coherence_notes=(
            "Counter-coherence manifests in the tension between structural gains and "
            "possible degradation in refusal robustness or consistency across tasks."
        ),
        artifact_ref_1=artifact_ref_1,
        artifact_ref_2=artifact_ref_2,
        next_iteration_plan=(
            "Increase prompt diversity, preserve matched baseline controls, and "
            "additional paired artifacts around the threshold neighborhood for a "
            "higher-confidence memo revision."
        ),
        metric_evidence=metric_evidence,
        qualitative_evidence=qualitative_pairs,
        figure_paths=figure_paths,
        confidence_limit=limit_conf,
        confidence_threshold=threshold_conf,
        confidence_bend=bend_conf,
        confidence_counter_coherence=counter_conf,
        edit_notes=edit_notes,
    )

    template_text = (Path("templates") / "theory_memo.md").read_text(encoding="utf-8")
    rendered = _render_template(template_text, context)
    rendered = rendered + _markdown_enrichment(context)

    out_path = run_dir / "theory_memo.md"
    out_path.write_text(rendered.strip() + "\n", encoding="utf-8")
    return out_path


def build_theory_memos_for_runs(runs_root: Path) -> list[Path]:
    """Build memos for each run-like folder under a root path."""
    runs_root = Path(runs_root)
    outputs: list[Path] = []
    for candidate in sorted(path for path in runs_root.glob("*") if path.is_dir()):
        if (candidate / "config.yaml").exists():
            outputs.append(build_theory_memo(candidate))
    return outputs
