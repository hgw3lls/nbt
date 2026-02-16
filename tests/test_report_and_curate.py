# ruff: noqa: I001
import json
from pathlib import Path

import pytest

CliRunner = pytest.importorskip("typer.testing").CliRunner
yaml = pytest.importorskip("yaml")

from neural_bending_toolkit.analysis.report import generate_markdown_report  # noqa: E402
from neural_bending_toolkit.cli import app  # noqa: E402


def _make_run_fixture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_for_report"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"experiment": "report-test", "name": "report-test"}),
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"metric_name": "kl_vs_baseline", "value": 0.44}),
                json.dumps({"metric_name": "entropy_delta", "value": 0.12}),
                json.dumps({"metric_name": "coherence_delta", "value": -0.05}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        json.dumps({"payload": {"attention": [[0.5, 0.5], [0.6, 0.4]]}}) + "\n",
        encoding="utf-8",
    )

    (artifacts / "sample_baseline.txt").write_text("baseline", encoding="utf-8")
    (artifacts / "sample_bent.txt").write_text("bent", encoding="utf-8")
    (artifacts / "sample.png").write_text("png", encoding="utf-8")
    return run_dir


def test_report_includes_bend_classification_section(tmp_path: Path) -> None:
    run_dir = _make_run_fixture(tmp_path)
    report_path = generate_markdown_report(run_dir)
    text = report_path.read_text(encoding="utf-8")

    assert "## Bend Classification" in text
    assert "Bend tag" in text
    assert "## Key Derived Metrics" in text
    assert "## Theory Memo" in text


def test_curate_exports_expected_files(tmp_path: Path) -> None:
    run_dir = _make_run_fixture(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["curate", str(run_dir)])

    assert result.exit_code == 0
    curated = run_dir / "curated"
    assert (curated / "theory_memo.md").exists()
    assert (curated / "bend_classification.json").exists()
    assert (curated / "derived_metrics.json").exists()
    assert (curated / "report.md").exists()
    assert (curated / "report.html").exists()
    assert (curated / "caption.md").exists()
