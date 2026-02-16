import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from neural_bending_toolkit.analysis.theory_memo_generator import (  # noqa: E402
    build_theory_memo,
)
from neural_bending_toolkit.registry import ExperimentRegistry  # noqa: E402
from neural_bending_toolkit.runner import run_experiment  # noqa: E402


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _fixture_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"experiment": "memo-test", "name": "memo-test"}),
        encoding="utf-8",
    )

    metrics = [
        {"metric_name": "kl_vs_baseline", "value": 0.61},
        {"metric_name": "entropy_delta", "value": 0.35},
        {"metric_name": "coherence_delta", "value": -0.22},
        {"metric_name": "structural_causality_delta", "value": 0.41},
        {"metric_name": "attractor_density_delta", "value": 0.33},
    ]
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in metrics) + "\n", encoding="utf-8"
    )

    _write_file(run_dir / "artifacts" / "sample_baseline.txt", "baseline text sample")
    _write_file(
        run_dir / "artifacts" / "sample_bent_divergence.txt", "bent text sample"
    )
    _write_file(run_dir / "analysis" / "figure_a.png", "png-placeholder")
    return run_dir


def test_memo_renders_with_required_sections_and_paragraphs(tmp_path: Path) -> None:
    run_dir = _fixture_run_dir(tmp_path)
    out_path = build_theory_memo(run_dir, seed=11)

    content = out_path.read_text(encoding="utf-8")
    required_headings = [
        "## Observed limits",
        "## Threshold behavior",
        "## Bend",
        "## Counter-coherence notes",
        "## Confidence by section",
    ]
    for heading in required_headings:
        assert heading in content

    assert "Limit → Threshold → Bend → Counter-Coherence" in content

    paragraphs = [
        line.strip()
        for line in content.splitlines()
        if len(line.strip()) > 60 and not line.strip().startswith("-")
    ]
    assert len(paragraphs) >= 4


def test_memo_generation_is_stable_for_fixed_seed(tmp_path: Path) -> None:
    run_dir = _fixture_run_dir(tmp_path)

    first = build_theory_memo(run_dir, seed=42).read_text(encoding="utf-8")
    second = build_theory_memo(run_dir, seed=42).read_text(encoding="utf-8")

    assert first == second


def test_auto_memo_generates_outputs_from_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "hello_config.yaml"
    config_path.write_text(
        yaml.safe_dump({"name": "Memo", "repeats": 1, "auto_memo": True}),
        encoding="utf-8",
    )

    registry = ExperimentRegistry()
    registry.discover_modules()

    from neural_bending_toolkit import runner as runner_module

    def _tmp_run_dir(experiment_name: str, runs_root: Path = Path("runs")) -> Path:
        run_dir = tmp_path / f"run_{experiment_name}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        return run_dir

    monkeypatch.setattr(runner_module, "build_run_dir", _tmp_run_dir)

    run_dir = run_experiment("hello-experiment", config_path, registry)

    assert (run_dir / "analysis" / "derived_metrics.json").exists()
    assert (run_dir / "analysis" / "bend_classification.json").exists()
    assert (run_dir / "theory_memo.md").exists()
