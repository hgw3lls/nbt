import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from neural_bending_toolkit.analysis.geopolitical_report import (
    compare_geopolitical_runs,
    generate_geopolitical_report,
)
from neural_bending_toolkit.cli import app


def _create_dummy_geopolitical_run(base_dir: Path, name: str) -> Path:
    run_dir = base_dir / name
    phase1 = run_dir / "artifacts" / "geopolitical" / "phase_1_ontology_mapping"
    phase2 = run_dir / "artifacts" / "geopolitical" / "phase_2_governance_dissonance"
    phase3 = run_dir / "artifacts" / "geopolitical" / "phase_3_justice_attractors"
    phase1.mkdir(parents=True)
    phase2.mkdir(parents=True)
    phase3.mkdir(parents=True)

    (phase1 / "dummy_similarity.csv").write_text(
        "concept,a,b\na,1.0,0.2\nb,0.2,1.0\n",
        encoding="utf-8",
    )
    np.save(phase1 / "dummy_pca.npy", np.array([[0.1, 0.2], [0.2, 0.1]]))
    np.save(phase1 / "dummy_umap.npy", np.array([[0.1, -0.1], [0.2, -0.2]]))

    phase2_rows = [
        {
            "pair_index": 1,
            "prompt_a": "A",
            "prompt_b": "B",
            "output_a": "response a",
            "output_b": "response b",
            "refusal_a": False,
            "refusal_b": True,
            "sentiment_a": -0.25,
            "sentiment_b": 0.5,
            "structural_score_a": 0.4,
            "structural_score_b": 0.7,
            "distribution_kl": 0.11,
        }
    ]
    (phase2 / "governance_dissonance_results.json").write_text(
        json.dumps(phase2_rows),
        encoding="utf-8",
    )

    phase3_rows = [
        {
            "token_set_index": 1,
            "token_set": ["equity", "rights"],
            "baseline_output": "baseline text",
            "bent_output": "bent text",
            "baseline_density": 0.05,
            "bent_density": 0.2,
            "density_change": 0.15,
        }
    ]
    (phase3 / "justice_attractor_results.json").write_text(
        json.dumps(phase3_rows),
        encoding="utf-8",
    )

    return run_dir


def test_generate_geopolitical_report_markdown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = _create_dummy_geopolitical_run(tmp_path, "run_a")

    report_path = generate_geopolitical_report(run_dir, format="markdown")

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "## Embedding Similarity Heatmaps" in content
    assert "## PCA/UMAP Plots" in content
    assert "## Sentiment Comparison" in content
    assert "## Refusal Rate Table" in content
    assert "## Attractor Density Comparison" in content


def test_generate_geopolitical_report_html(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = _create_dummy_geopolitical_run(tmp_path, "run_b")

    report_path = generate_geopolitical_report(run_dir, format="html")

    assert report_path.exists()
    assert report_path.suffix == ".html"


def test_compare_geopolitical_runs_outputs_csv_and_markdown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    run1 = _create_dummy_geopolitical_run(tmp_path, "run_1")
    run2 = _create_dummy_geopolitical_run(tmp_path, "run_2")

    csv_path, md_path = compare_geopolitical_runs([run1, run2])

    assert csv_path.exists()
    assert md_path.exists()
    assert "Refusal Rate" in md_path.read_text(encoding="utf-8")


def test_cli_geopolitical_report_and_compare(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run1 = _create_dummy_geopolitical_run(tmp_path, "run_cli_1")
    run2 = _create_dummy_geopolitical_run(tmp_path, "run_cli_2")
    runner = CliRunner()

    report_result = runner.invoke(app, ["geopolitical", "report", str(run1)])
    compare_result = runner.invoke(
        app,
        ["geopolitical", "compare", str(run1), str(run2)],
    )

    assert report_result.exit_code == 0
    assert "Geopolitical report generated" in report_result.stdout
    assert compare_result.exit_code == 0
    assert "Geopolitical comparison CSV" in compare_result.stdout
