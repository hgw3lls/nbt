from __future__ import annotations

import json
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from typer.testing import CliRunner

from neural_bending_toolkit.cli import app
from neural_bending_toolkit.figures.builder import (
    build_figure_from_spec,
    build_figures_from_run,
)
from neural_bending_toolkit.figures.specs import FigureSpec, load_figure_spec


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _dummy_run(tmp_path: Path, name: str = "run_a") -> Path:
    run = tmp_path / "runs" / name
    phase1 = run / "artifacts" / "geopolitical" / "phase_1_ontology_mapping"
    phase2 = run / "artifacts" / "geopolitical" / "phase_2_governance_dissonance"
    phase3 = run / "artifacts" / "geopolitical" / "phase_3_justice_attractors"
    phase1.mkdir(parents=True, exist_ok=True)
    phase2.mkdir(parents=True, exist_ok=True)
    phase3.mkdir(parents=True, exist_ok=True)

    (phase1 / "dummy_similarity.csv").write_text(
        "concept,c1,c2\nc1,1.0,0.2\nc2,0.2,1.0\n",
        encoding="utf-8",
    )
    np.save(phase1 / "dummy_umap.npy", np.array([[0.1, 0.2], [0.3, 0.4]]))

    phase2_rows = [
        {
            "refusal_a": False,
            "refusal_b": True,
            "sentiment_a": -0.1,
            "sentiment_b": 0.2,
            "structural_score_a": 0.7,
            "structural_score_b": 0.4,
        }
    ]
    (phase2 / "governance_dissonance_results.json").write_text(
        json.dumps(phase2_rows),
        encoding="utf-8",
    )

    phase3_rows = [
        {
            "baseline_density": 0.1,
            "bent_density": 0.3,
            "density_change": 0.2,
        }
    ]
    (phase3 / "justice_attractor_results.json").write_text(
        json.dumps(phase3_rows),
        encoding="utf-8",
    )

    _write_jsonl(
        run / "metrics.jsonl",
        [
            {"metric_name": "distribution_kl", "value": 0.3},
            {"metric_name": "js_divergence", "value": 0.2},
        ],
    )
    _write_jsonl(
        run / "events.jsonl",
        [{"payload": {"attention": [[0.2, 0.8], [0.4, 0.6]]}}],
    )

    # dummy image artifact for montage
    img = np.zeros((12, 12, 3), dtype=np.float32)
    img[..., 1] = 1.0
    plt.imsave(run / "artifacts" / "dummy.png", img)

    # dummy audio artifact for montage fallback
    audio = (np.sin(np.linspace(0.0, 20.0, 800)) * 20000).astype(np.int16)
    wav_path = run / "artifacts" / "dummy.wav"
    with wave.open(str(wav_path), "wb") as wavf:
        wavf.setnchannels(1)
        wavf.setsampwidth(2)
        wavf.setframerate(16000)
        wavf.writeframes(audio.tobytes())

    return run


def _write_spec(tmp_path: Path, run_dir: Path, figure_id: str, plot_type: str) -> Path:
    spec = {
        "figure_id": figure_id,
        "title": figure_id,
        "input_run_dirs": [str(run_dir)],
        "plot_type": plot_type,
        "inputs": {},
        "output_format": {"png": True, "pdf": True},
        "caption_template_variables": {"limit": "x", "threshold": "y"},
    }
    # plot-specific inputs
    if plot_type == "embedding_similarity_heatmap":
        spec["inputs"] = {
            "similarity_csv": (
                "artifacts/geopolitical/phase_1_ontology_mapping/" "*_similarity.csv"
            )
        }
    elif plot_type == "embedding_umap_scatter":
        spec["inputs"] = {
            "umap_npy": "artifacts/geopolitical/phase_1_ontology_mapping/*_umap.npy"
        }
    elif plot_type in {
        "refusal_rate_table_to_figure",
        "causal_framing_bar_chart",
    }:
        spec["inputs"] = {
            "phase2_json": (
                "artifacts/geopolitical/phase_2_governance_dissonance/"
                "governance_dissonance_results.json"
            )
        }
    elif plot_type == "attractor_density_comparison":
        spec["inputs"] = {
            "phase3_json": (
                "artifacts/geopolitical/phase_3_justice_attractors/"
                "justice_attractor_results.json"
            )
        }

    path = tmp_path / f"{figure_id}.yaml"
    path.write_text(json.dumps(spec), encoding="utf-8")
    # json is valid yaml
    return path


def test_figure_spec_validation() -> None:
    good = FigureSpec.model_validate(
        {
            "figure_id": "fig_a",
            "title": "Figure A",
            "input_run_dirs": ["runs/*"],
            "plot_type": "divergence_bar_chart",
            "inputs": {},
            "output_format": {"png": True, "pdf": True},
            "caption_template_variables": {},
        }
    )
    assert good.figure_id == "fig_a"


def test_each_plot_type_builds_on_dummy_data(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = _dummy_run(tmp_path)
    plot_types = [
        "embedding_similarity_heatmap",
        "embedding_umap_scatter",
        "attention_entropy_timeseries",
        "divergence_bar_chart",
        "refusal_rate_table_to_figure",
        "causal_framing_bar_chart",
        "attractor_density_comparison",
        "montage_grid",
    ]
    for idx, plot_type in enumerate(plot_types, start=1):
        spec_path = _write_spec(tmp_path, run, f"fig_{idx}", plot_type)
        output_dir = build_figure_from_spec(spec_path, repo_root=tmp_path)
        assert (output_dir / f"fig_{idx}.png").exists()
        assert (output_dir / f"fig_{idx}.pdf").exists()
        assert (output_dir / "caption.md").exists()
        assert (output_dir / "provenance.json").exists()


def test_build_from_run_finds_specs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = _dummy_run(tmp_path, "run_with_specs")
    specs_dir = run / "figure_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    first = _write_spec(specs_dir, run, "from_run_1", "divergence_bar_chart")
    first.rename(specs_dir / "from_run_1.yaml")
    second = _write_spec(specs_dir, run, "from_run_2", "refusal_rate_table_to_figure")
    second.rename(specs_dir / "from_run_2.yaml")

    outputs = build_figures_from_run(run, repo_root=tmp_path)
    assert len(outputs) == 2


def test_cli_figure_build_and_build_from_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = _dummy_run(tmp_path, "run_cli")
    spec = _write_spec(tmp_path, run, "cli_fig", "divergence_bar_chart")

    runner = CliRunner()
    result = runner.invoke(app, ["figure", "build", "--spec", str(spec)])
    assert result.exit_code == 0

    specs_dir = run / "figure_specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    spec_in_run = _write_spec(
        specs_dir, run, "cli_from_run", "attractor_density_comparison"
    )
    spec_in_run.rename(specs_dir / "cli_from_run.yaml")

    result2 = runner.invoke(app, ["figure", "build-from-run", str(run)])
    assert result2.exit_code == 0
    assert "Built" in result2.stdout


def test_load_figure_spec_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "spec.yaml"
    path.write_text(
        """
figure_id: roundtrip
title: Roundtrip
input_run_dirs: [runs/*]
plot_type: divergence_bar_chart
inputs: {}
output_format: {png: true, pdf: true}
caption_template_variables: {limit: drift, threshold: spike}
""",
        encoding="utf-8",
    )
    spec = load_figure_spec(path)
    assert spec.figure_id == "roundtrip"
