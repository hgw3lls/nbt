import sys
import types

if "matplotlib" not in sys.modules:
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

yaml = pytest.importorskip("yaml")

from neural_bending_toolkit.analysis.bend_classifier import (  # noqa: E402
    BendClassification,
    classify_bend,
)
from neural_bending_toolkit.analysis.derived_metrics import (  # noqa: E402
    DerivedMetrics,
    write_derived_metrics,
)
from neural_bending_toolkit.cli import app  # noqa: E402


def _write_run_dir(
    run_dir: Path, metrics: list[dict[str, float]], config: dict
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in metrics) + "\n",
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def test_derived_metrics_schema_minimal_fixture(tmp_path: Path) -> None:
    run_dir = tmp_path / "minimal_run"
    _write_run_dir(
        run_dir,
        metrics=[
            {"metric_name": "kl_vs_baseline", "value": 0.42},
            {"metric_name": "refusal_rate", "value": 0.13},
            {"metric_name": "baseline_refusal_rate", "value": 0.10},
            {"metric_name": "structural_causality", "value": 0.71},
            {"metric_name": "baseline_structural_causality", "value": 0.52},
        ],
        config={"experiment": "dummy"},
    )

    out_path = write_derived_metrics(run_dir)
    payload = DerivedMetrics.model_validate_json(out_path.read_text(encoding="utf-8"))

    assert out_path == run_dir / "analysis" / "derived_metrics.json"
    assert payload.divergence == 0.42
    assert payload.refusal_rate_delta == 0.03
    assert payload.structural_causality_delta == pytest.approx(0.19)
    assert "divergence" in payload.availability
    assert "divergence" in payload.normalized


def test_classifier_tag_logic_three_cases(tmp_path: Path) -> None:
    revelatory_dir = tmp_path / "revelatory_case"
    _write_run_dir(
        revelatory_dir,
        metrics=[
            {"metric_name": "kl_vs_baseline", "value": 0.20},
            {"metric_name": "attractor_density", "value": 0.62},
            {"metric_name": "baseline_attractor_density", "value": 0.40},
            {"metric_name": "structural_causality", "value": 0.80},
            {"metric_name": "baseline_structural_causality", "value": 0.45},
            {"metric_name": "cross_task_consistency", "value": 0.75},
            {"metric_name": "refusal_rate", "value": 0.05},
            {"metric_name": "baseline_refusal_rate", "value": 0.05},
        ],
        config={"experiment": "revelatory"},
    )
    revelatory = classify_bend(revelatory_dir)

    disruptive_dir = tmp_path / "disruptive_case"
    _write_run_dir(
        disruptive_dir,
        metrics=[
            {"metric_name": "kl_vs_baseline", "value": 0.95},
            {"metric_name": "entropy_delta", "value": 0.60},
            {"metric_name": "coherence_delta", "value": -0.45},
            {"metric_name": "refusal_rate", "value": 0.40},
            {"metric_name": "baseline_refusal_rate", "value": 0.10},
            {"metric_name": "structural_causality_delta", "value": -0.10},
        ],
        config={"experiment": "disruptive"},
    )
    disruptive = classify_bend(disruptive_dir)

    recoherent_dir = tmp_path / "recoherent_case"
    _write_run_dir(
        recoherent_dir,
        metrics=[
            {"metric_name": "kl_vs_baseline", "value": 0.08},
            {"metric_name": "coherence_delta", "value": 0.55},
            {"metric_name": "entropy_delta", "value": -0.30},
            {"metric_name": "refusal_rate", "value": 0.08},
            {"metric_name": "baseline_refusal_rate", "value": 0.18},
            {"metric_name": "structural_causality_delta", "value": 0.12},
            {"metric_name": "cross_task_consistency", "value": 0.40},
        ],
        config={"experiment": "recoherent"},
    )
    recoherent = classify_bend(recoherent_dir)

    assert revelatory.bend_tag == "revelatory"
    assert disruptive.bend_tag == "disruptive"
    assert recoherent.bend_tag == "recoherent"


def test_cli_analyze_all_writes_both_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "cli_case"
    _write_run_dir(
        run_dir,
        metrics=[
            {"metric_name": "kl_vs_baseline", "value": 0.2},
            {"metric_name": "coherence_delta", "value": 0.1},
        ],
        config={
            "geopolitical": {"models": ["m1", "m2"], "prompt_pairs": ["p1", "p2"]}
        },
    )

    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "all", str(run_dir)])

    assert result.exit_code == 0
    derived = DerivedMetrics.model_validate_json(
        (run_dir / "analysis" / "derived_metrics.json").read_text(encoding="utf-8")
    )
    classification = BendClassification.model_validate_json(
        (run_dir / "analysis" / "bend_classification.json").read_text(
            encoding="utf-8"
        )
    )
    assert derived.divergence == 0.2
    assert classification.geopolitical_flag is True


def test_bend_v2_trace_metrics_written_to_metrics_and_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "trace_case"
    _write_run_dir(
        run_dir,
        metrics=[
            {"step": 0, "metric_name": "attention_entropy", "value": 0.9},
            {"step": 0, "metric_name": "attention_entropy", "value": 1.1},
            {"step": 1, "metric_name": "attention_entropy", "value": 0.6},
            {"step": 1, "metric_name": "attention_entropy", "value": 0.8},
            {"step": 0, "metric_name": "attention_topk_mass", "value": 0.55},
            {"step": 1, "metric_name": "attention_topk_mass", "value": 0.65},
        ],
        config={"experiment": "trace"},
    )

    out_path = write_derived_metrics(run_dir)
    payload = DerivedMetrics.model_validate_json(out_path.read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metric_rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    names = {str(row.get("metric_name", "")) for row in metric_rows}

    assert payload.attention_entropy_mean_per_step == pytest.approx(0.85)
    assert payload.attention_entropy_variance_per_step == pytest.approx(0.01)
    assert payload.attention_concentration_topk_mass == pytest.approx(0.60)
    assert payload.attention_entropy_recovery_index == pytest.approx(0.3)
    assert "bend_v2_trace_metrics" in summary
    assert "attention_entropy_mean_per_step" in names
    assert "attention_entropy_variance_per_step" in names
    assert "attention_concentration_topk_mass" in names
    assert "attention_entropy_recovery_index" in names
